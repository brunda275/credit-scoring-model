import argparse, json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# XGBoost wrapper that encodes y to ints internally
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class XGBLabelEncoded(XGBClassifier):
    def fit(self, X, y, **kwargs):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        if getattr(self, "objective", None) is None:
            if len(self.le_.classes_) <= 2:
                self.set_params(objective="binary:logistic")
            else:
                self.set_params(objective="multi:softprob", num_class=len(self.le_.classes_))
        out = super().fit(X, y_enc, **kwargs)
        self.original_classes_ = self.le_.classes_
        return out
    def predict(self, X, **kwargs):
        y_enc = super().predict(X, **kwargs)
        return self.le_.inverse_transform(y_enc.astype(int))
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)

def to_dense_array(X):
    return X.toarray() if hasattr(X, "toarray") else X

def main(args):
    df = pd.read_csv(args.data)
    target = args.target

    # Drop obvious IDs
    id_like = [c for c in df.columns if any(x in c.lower() for x in ["id","customer","serial","ssn"])]
    df = df[[c for c in df.columns if c not in id_like]]

    X = df.drop(columns=[target])
    y = df[target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ],
        remainder="drop",
        sparse_threshold=1.0 # still returns sparse if OHE sparse
    )

    to_dense = FunctionTransformer(to_dense_array, accept_sparse=True)

    def make_pipe(model):
        steps = [("prep", preprocess)]
        if args.smote.lower() in ("on","true","1","yes"):
            steps += [("to_dense", to_dense), ("smote", SMOTE(random_state=42))]
        steps += [("model", model)]
        return ImbPipeline(steps)

    models = {
        "LogReg": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
        "XGBoost": XGBLabelEncoded(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist"
        ),
    }

    rows = []
    for name, mdl in models.items():
        pipe = make_pipe(mdl)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)

        # get class order for metrics
        model_in = pipe.named_steps["model"]
        classes_ = getattr(model_in, "original_classes_", getattr(model_in, "classes_", None))
        y_test_bin = label_binarize(y_test, classes=classes_)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision(macro)": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall(macro)": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1(macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            "ROC-AUC(ovr,macro)": roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr"),
        }
        rows.append((name, pipe, metrics))

    # choose best by F1
    rows.sort(key=lambda r: r[2]["F1(macro)"], reverse=True)
    best_name, best_pipe, best_metrics = rows[0]

    print("\n=== RESULTS ===")
    for _, _, m in rows: print(m)
    print("\nBest:", best_name, best_metrics)

    joblib.dump(best_pipe, "credit_scoring_model.joblib")

    # also save metrics
    with open("metrics.json","w") as f:
        json.dump({"results":[r[2] for r in rows], "best":best_metrics}, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV")
    p.add_argument("--target", required=True, help="Target column (e.g., Risk_good)")
    p.add_argument("--smote", default="on", help="on/off")
    args = p.parse_args()
    main(args)
