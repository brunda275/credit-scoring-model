'''# ---------- app.py (high-contrast dark theme, full width) ----------

# Needed by the saved pipeline (used inside FunctionTransformer)
def to_dense_array(X):
    return X.toarray() if hasattr(X, "toarray") else X

# If your best model was XGBoost, the pickle may reference this class.
# Keeping it here makes unpickling robust. (Harmless if not used.)
try:
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
except Exception:
    XGBLabelEncoded = None  # noqa

import os
import json
import joblib
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

# ---- Load model ----
MODEL_PATH = os.environ.get("MODEL_PATH", "credit_scoring_model.joblib")
print(f"Loading model from {MODEL_PATH} ...")
pipe = joblib.load(MODEL_PATH)

# Robust classes getter (works for XGBoost wrapper & others)
model_in_pipe = pipe.named_steps["model"]
_cls = getattr(model_in_pipe, "original_classes_", getattr(model_in_pipe, "classes_", None))
MODEL_CLASSES = list(_cls)

# Columns (for UI)
NUM_COLS = ["age","income","debt","years_employed","number_of_loans","credit_utilization","late_payments_12m"]
CAT_COLS = ["home_ownership","education","payment_history","has_mortgage","region"]

# Default dropdown choices (refine from CSV if present)
DEFAULT_CHOICES = {
    "home_ownership": ["RENT","MORTGAGE","OWN"],
    "education": ["HighSchool","Graduate","Postgraduate"],
    "payment_history": ["OnTime","Missed","Defaulted"],
    "has_mortgage": ["Yes","No"],
    "region": ["North","South","East","West"],
}
if os.path.exists("credit_scoring_sample.csv"):
    try:
        _df = pd.read_csv("credit_scoring_sample.csv")
        for col in CAT_COLS:
            if col in _df.columns:
                DEFAULT_CHOICES[col] = sorted(map(str, _df[col].dropna().astype(str).unique()))
    except Exception:
        pass

# ---- Inference helpers ----
def predict_json(record: dict):
    df_in = pd.DataFrame([record])
    proba = pipe.predict_proba(df_in)[0]
    pred = pipe.predict(df_in)[0]
    return {
        "prediction": str(pred),
        "probabilities": {cls: float(p) for cls, p in zip(MODEL_CLASSES, proba)},
        "classes": MODEL_CLASSES
    }

def risk_score_from_proba(probs):
    # Poor=0, Standard=0.5, Good=1.0
    weights = {"Poor":0.0, "Standard":0.5, "Good":1.0}
    return round(100*sum(weights[k]*float(v) for k,v in probs.items()), 2)

def risk_band(score):
    if score >= 70: return "Low Risk"
    if score >= 40: return "Medium Risk"
    return "High Risk"

def risk_bar(score):
    # horizontal progress bar 0..100 using matplotlib
    fig, ax = plt.subplots(figsize=(5.5, 0.9))
    ax.barh([0], [score/100], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0","25","50","75","100"])
    ax.text(min(score/100,0.98), 0, f"{score:.1f}", va="center",
            ha="right" if score/100>0.15 else "left")
    fig.tight_layout()
    return fig

def prob_bar_figure(probs):
    classes = list(probs.keys())
    values = [float(probs[c]) for c in classes]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(classes, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig

def extra_stats_md(record, score, probs):
    dti = (record["debt"] / (record["income"]+1)) if record["income"] else np.nan
    util = record["credit_utilization"]/100.0
    lp = record["late_payments_12m"]
    band = risk_band(score)
    md = f"""
### Prediction: **{max(probs, key=probs.get)}**  ·  **Risk Score:** **{score:.1f} / 100**  ·  **{band}**
**Debt-to-Income:** {dti:.2f}   ·   **Credit Utilization:** {util:.2f}  
**Late payments (12m):** {lp}   ·   **Loans:** {record["number_of_loans"]}
"""
    return md

def run_predict(age, income, debt, years_employed, number_of_loans,
                credit_utilization, late_payments_12m,
                home_ownership, education, payment_history, has_mortgage, region):
    rec = {
        "age": age, "income": income, "debt": debt, "years_employed": years_employed,
        "number_of_loans": number_of_loans, "credit_utilization": credit_utilization,
        "late_payments_12m": late_payments_12m, "home_ownership": home_ownership,
        "education": education, "payment_history": payment_history,
        "has_mortgage": has_mortgage, "region": region,
    }
    out = predict_json(rec)
    score = risk_score_from_proba(out["probabilities"])
    pred_md = extra_stats_md(rec, score, out["probabilities"])
    fig_score = risk_bar(score)
    fig_probs = prob_bar_figure(out["probabilities"])
    return pred_md, fig_score, fig_probs, out

# ---- Optional: read training metrics if present ----
def model_info_text():
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json","r") as f:
                m = json.load(f)
            best = m.get("best", {})
            return (
                f"**Best model**: {best.get('Model','?')}  \n"
                f"**F1 (macro)**: {best.get('F1(macro)',0):.3f}  ·  "
                f"**Accuracy**: {best.get('Accuracy',0):.3f}  ·  "
                f"**ROC-AUC (macro, OvR)**: {best.get('ROC-AUC(ovr,macro)',0):.3f}"
            )
        except Exception:
            return "Metrics file found but could not be parsed."
    return "No `metrics.json` found yet. Train locally with `python train.py --data credit_scoring_sample.csv --target Risk_good`."

# ---- Theme & CSS (dark, readable, full-width) ----
theme = gr.themes.Default()  # base; we override via CSS
css = """
:root, body, .gradio-container { 
  background: #0e121b !important;     /* dark navy background */
  color: #e6eaf2 !important;          /* light text */
}
.gradio-container { max-width: 1440px !important; }

/* inputs & dropdowns readable on dark */
input, select, textarea { 
  background: #0f1522 !important; 
  color: #e6eaf2 !important; 
  border-color: #2b3447 !important; 
}
.gr-number input, .gr-textbox input, .gr-textbox textarea, 
.gr-dropdown input, .gr-dropdown .wrap { 
  background: #0f1522 !important; 
  color: #e6eaf2 !important; 
  border-color: #2b3447 !important; 
}

/* section/card containers */
.section { 
  background: #121826 !important; 
  border: 1px solid #2b3447 !important; 
  border-radius: 14px; 
  padding: 12px 14px; 
  color: #e6eaf2 !important; 
}

/* result card */
#pred-card { 
  background: #141c2b !important; 
  border: 1px solid #33415e !important; 
  border-radius: 16px; 
  padding: 16px 18px; 
  color: #ffffff !important; 
  box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
#pred-card * { color: #ffffff !important; }

/* plot title bars and labels */
.label, .panel .label, .plot .label { 
  color: #e6eaf2 !important; 
  background: #0f1522 !important; 
  border-color: #2b3447 !important; 
}

footer, .footer { display: none !important; }
"""

# ---- UI ----
with gr.Blocks(title="Credit Scoring Model", theme=theme, css=css, fill_height=True) as demo:
    gr.Markdown("## 💳 Credit Scoring Model")
    with gr.Row(equal_height=True):
        # LEFT: Inputs
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["section"]):
                gr.Markdown("#### Inputs")
                age = gr.Number(label="age", value=30)
                income = gr.Number(label="income", value=72000)
                debt = gr.Number(label="debt", value=12000)
                years_employed = gr.Number(label="years_employed", value=5.0)
                number_of_loans = gr.Number(label="number_of_loans", value=2)
                credit_utilization = gr.Number(label="credit_utilization (0–100%)", value=35.0)
                late_payments_12m = gr.Number(label="late_payments_12m", value=0)
                home_ownership = gr.Dropdown(DEFAULT_CHOICES["home_ownership"], value=DEFAULT_CHOICES["home_ownership"][0], label="home_ownership")
                education = gr.Dropdown(DEFAULT_CHOICES["education"], value=DEFAULT_CHOICES["education"][0], label="education")
                payment_history = gr.Dropdown(DEFAULT_CHOICES["payment_history"], value=DEFAULT_CHOICES["payment_history"][0], label="payment_history")
                has_mortgage = gr.Dropdown(DEFAULT_CHOICES["has_mortgage"], value=DEFAULT_CHOICES["has_mortgage"][1], label="has_mortgage")
                region = gr.Dropdown(DEFAULT_CHOICES["region"], value=DEFAULT_CHOICES["region"][1], label="region")

                with gr.Row():
                    sample_btn = gr.Button("🎯 Use Sample")
                    clear_btn = gr.Button("🧹 Clear")

                # Fill sample & clear handlers
                sample_values = [30, 72000, 12000, 5.0, 2, 35.0, 0, "RENT", "Graduate", "OnTime", "No", "South"]
                sample_btn.click(lambda: sample_values, outputs=[
                    age, income, debt, years_employed, number_of_loans,
                    credit_utilization, late_payments_12m, home_ownership, education,
                    payment_history, has_mortgage, region
                ])
                clear_btn.click(lambda: [None]*12, outputs=[
                    age, income, debt, years_employed, number_of_loans,
                    credit_utilization, late_payments_12m, home_ownership, education,
                    payment_history, has_mortgage, region
                ])

                predict_btn = gr.Button("🔮 Predict", variant="primary")

        # RIGHT: Results
        with gr.Column(scale=5):
            pred_md = gr.Markdown(elem_id="pred-card")
            score_plot = gr.Plot(label="Risk Score")
            prob_plot = gr.Plot(label="Class Probabilities")
            json_out = gr.JSON(label="Raw output")

    # Model info / training metrics
    with gr.Accordion("Model Info", open=False):
        gr.Markdown(model_info_text())

    inputs = [age, income, debt, years_employed, number_of_loans,
              credit_utilization, late_payments_12m,
              home_ownership, education, payment_history, has_mortgage, region]
    predict_btn.click(run_predict, inputs=inputs, outputs=[pred_md, score_plot, prob_plot, json_out])

if __name__ == "__main__":
    print("Model loaded. Launching Gradio on http://127.0.0.1:7860 ...")
    demo.launch(
        server_name="127.0.0.1",   # use "0.0.0.0" if you need external access
        server_port=7860,          # change if port is busy
        inbrowser=True,            # auto-open browser
        share=False,
        show_error=True
    )
# ---------- end app.py ----------
'''
# ---------- app.py (dark theme + polished input boxes) ----------

# Needed by the saved pipeline (used inside FunctionTransformer)
def to_dense_array(X):
    return X.toarray() if hasattr(X, "toarray") else X

# If your best model was XGBoost, the pickle may reference this class.
# Keeping it here makes unpickling robust. (Harmless if not used.)
try:
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
except Exception:
    XGBLabelEncoded = None  # noqa

import os
import json
import joblib
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

# ---- Load model ----
MODEL_PATH = os.environ.get("MODEL_PATH", "credit_scoring_model.joblib")
print(f"Loading model from {MODEL_PATH} ...")
pipe = joblib.load(MODEL_PATH)

# Robust classes getter (works for XGBoost wrapper & others)
model_in_pipe = pipe.named_steps["model"]
_cls = getattr(model_in_pipe, "original_classes_", getattr(model_in_pipe, "classes_", None))
MODEL_CLASSES = list(_cls)

# Columns (for UI)
NUM_COLS = ["age","income","debt","years_employed","number_of_loans","credit_utilization","late_payments_12m"]
CAT_COLS = ["home_ownership","education","payment_history","has_mortgage","region"]

# Default dropdown choices (refine from CSV if present)
DEFAULT_CHOICES = {
    "home_ownership": ["RENT","MORTGAGE","OWN"],
    "education": ["HighSchool","Graduate","Postgraduate"],
    "payment_history": ["OnTime","Missed","Defaulted"],
    "has_mortgage": ["Yes","No"],
    "region": ["North","South","East","West"],
}
if os.path.exists("credit_scoring_sample.csv"):
    try:
        _df = pd.read_csv("credit_scoring_sample.csv")
        for col in CAT_COLS:
            if col in _df.columns:
                DEFAULT_CHOICES[col] = sorted(map(str, _df[col].dropna().astype(str).unique()))
    except Exception:
        pass

# ---- Inference helpers ----
def predict_json(record: dict):
    df_in = pd.DataFrame([record])
    proba = pipe.predict_proba(df_in)[0]
    pred = pipe.predict(df_in)[0]
    return {
        "prediction": str(pred),
        "probabilities": {cls: float(p) for cls, p in zip(MODEL_CLASSES, proba)},
        "classes": MODEL_CLASSES
    }

def risk_score_from_proba(probs):
    # Poor=0, Standard=0.5, Good=1.0
    weights = {"Poor":0.0, "Standard":0.5, "Good":1.0}
    return round(100*sum(weights[k]*float(v) for k,v in probs.items()), 2)

def risk_band(score):
    if score >= 70: return "Low Risk"
    if score >= 40: return "Medium Risk"
    return "High Risk"

def risk_bar(score):
    # horizontal progress bar 0..100 using matplotlib
    fig, ax = plt.subplots(figsize=(5.5, 0.9))
    ax.barh([0], [score/100], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0","25","50","75","100"])
    ax.text(min(score/100,0.98), 0, f"{score:.1f}", va="center",
            ha="right" if score/100>0.15 else "left")
    fig.tight_layout()
    return fig

def prob_bar_figure(probs):
    classes = list(probs.keys())
    values = [float(probs[c]) for c in classes]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(classes, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig

def extra_stats_md(record, score, probs):
    dti = (record["debt"] / (record["income"]+1)) if record["income"] else np.nan
    util = record["credit_utilization"]/100.0
    lp = record["late_payments_12m"]
    band = risk_band(score)
    md = f"""
### Prediction: **{max(probs, key=probs.get)}**  ·  **Risk Score:** **{score:.1f} / 100**  ·  **{band}**
**Debt-to-Income:** {dti:.2f}   ·   **Credit Utilization:** {util:.2f}  
**Late payments (12m):** {lp}   ·   **Loans:** {record["number_of_loans"]}
"""
    return md

def run_predict(age, income, debt, years_employed, number_of_loans,
                credit_utilization, late_payments_12m,
                home_ownership, education, payment_history, has_mortgage, region):
    rec = {
        "age": age, "income": income, "debt": debt, "years_employed": years_employed,
        "number_of_loans": number_of_loans, "credit_utilization": credit_utilization,
        "late_payments_12m": late_payments_12m, "home_ownership": home_ownership,
        "education": education, "payment_history": payment_history,
        "has_mortgage": has_mortgage, "region": region,
    }
    out = predict_json(rec)
    score = risk_score_from_proba(out["probabilities"])
    pred_md = extra_stats_md(rec, score, out["probabilities"])
    fig_score = risk_bar(score)
    fig_probs = prob_bar_figure(out["probabilities"])
    return pred_md, fig_score, fig_probs, out

# ---- Optional: read training metrics if present ----
def model_info_text():
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json","r") as f:
                m = json.load(f)
            best = m.get("best", {})
            return (
                f"**Best model**: {best.get('Model','?')}  \n"
                f"**F1 (macro)**: {best.get('F1(macro)',0):.3f}  ·  "
                f"**Accuracy**: {best.get('Accuracy',0):.3f}  ·  "
                f"**ROC-AUC (macro, OvR)**: {best.get('ROC-AUC(ovr,macro)',0):.3f}"
            )
        except Exception:
            return "Metrics file found but could not be parsed."
    return "No `metrics.json` found yet. Train locally with `python train.py --data credit_scoring_sample.csv --target Risk_good`."

# ---- Theme & CSS (dark, readable, full-width, styled inputs) ----
theme = gr.themes.Default()  # base; we override via CSS
css = """
:root, body, .gradio-container { 
  background: #0e121b !important;     /* dark navy background */
  color: #e6eaf2 !important;          /* light text */
}
.gradio-container { max-width: 1440px !important; }

/* section/card containers */
.section { 
  background: #121826 !important; 
  border: 1px solid #2b3447 !important; 
  border-radius: 14px; 
  padding: 12px 14px; 
  color: #e6eaf2 !important; 
}

/* --- INPUT COLUMN POLISH --- */

/* Remove default dark blocks around inputs inside .section */
.section .gr-box,
.section .gr-panel,
.section .gr-form { 
  background: transparent !important; 
  border: none !important; 
  box-shadow: none !important; 
}

/* Actual input fields (Numbers, Textboxes, Dropdowns) */
.inp input,
.inp textarea,
.inp .wrap,
.inp .wrap input {
  background: #10192e !important;           /* soft navy field */
  color: #e6eaf2 !important;                 /* light text */
  border: 1px solid #36425e !important;      /* subtle border */
  border-radius: 12px !important;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,.15) !important;
}

/* focus ring */
.inp input:focus,
.inp textarea:focus,
.inp .wrap:focus-within,
.inp .wrap input:focus {
  border-color: #5aa2ff !important;
  box-shadow: 0 0 0 2px rgba(90,162,255,.25) !important;
  outline: none !important;
}

/* labels on dark */
.label, .panel .label, .plot .label { 
  color: #e6eaf2 !important; 
  background: #0f1522 !important; 
  border-color: #2b3447 !important; 
}

/* result card */
#pred-card { 
  background: #141c2b !important; 
  border: 1px solid #33415e !important; 
  border-radius: 16px; 
  padding: 16px 18px; 
  color: #ffffff !important; 
  box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
#pred-card * { color: #ffffff !important; }

footer, .footer { display: none !important; }
"""

# ---- UI ----
with gr.Blocks(title="Credit Scoring Model", theme=theme, css=css, fill_height=True) as demo:
    gr.Markdown("## 💳 Credit Scoring Model")
    with gr.Row(equal_height=True):
        # LEFT: Inputs
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["section"]):
                gr.Markdown("#### Inputs")
                age = gr.Number(label="age", value=30, elem_classes=["inp"])
                income = gr.Number(label="income", value=72000, elem_classes=["inp"])
                debt = gr.Number(label="debt", value=12000, elem_classes=["inp"])
                years_employed = gr.Number(label="years_employed", value=5.0, elem_classes=["inp"])
                number_of_loans = gr.Number(label="number_of_loans", value=2, elem_classes=["inp"])
                credit_utilization = gr.Number(label="credit_utilization (0–100%)", value=35.0, elem_classes=["inp"])
                late_payments_12m = gr.Number(label="late_payments_12m", value=0, elem_classes=["inp"])
                home_ownership = gr.Dropdown(DEFAULT_CHOICES["home_ownership"], value=DEFAULT_CHOICES["home_ownership"][0], label="home_ownership", elem_classes=["inp"])
                education = gr.Dropdown(DEFAULT_CHOICES["education"], value=DEFAULT_CHOICES["education"][0], label="education", elem_classes=["inp"])
                payment_history = gr.Dropdown(DEFAULT_CHOICES["payment_history"], value=DEFAULT_CHOICES["payment_history"][0], label="payment_history", elem_classes=["inp"])
                has_mortgage = gr.Dropdown(DEFAULT_CHOICES["has_mortgage"], value=DEFAULT_CHOICES["has_mortgage"][1], label="has_mortgage", elem_classes=["inp"])
                region = gr.Dropdown(DEFAULT_CHOICES["region"], value=DEFAULT_CHOICES["region"][1], label="region", elem_classes=["inp"])

                with gr.Row():
                    sample_btn = gr.Button("🎯 Use Sample")
                    clear_btn = gr.Button("🧹 Clear")

                # Fill sample & clear handlers
                sample_values = [30, 72000, 12000, 5.0, 2, 35.0, 0, "RENT", "Graduate", "OnTime", "No", "South"]
                sample_btn.click(lambda: sample_values, outputs=[
                    age, income, debt, years_employed, number_of_loans,
                    credit_utilization, late_payments_12m, home_ownership, education,
                    payment_history, has_mortgage, region
                ])
                clear_btn.click(lambda: [None]*12, outputs=[
                    age, income, debt, years_employed, number_of_loans,
                    credit_utilization, late_payments_12m, home_ownership, education,
                    payment_history, has_mortgage, region
                ])

                predict_btn = gr.Button("🔮 Predict", variant="primary")

        # RIGHT: Results
        with gr.Column(scale=5):
            pred_md = gr.Markdown(elem_id="pred-card")
            score_plot = gr.Plot(label="Risk Score")
            prob_plot = gr.Plot(label="Class Probabilities")
            json_out = gr.JSON(label="Raw output")

    # Model info / training metrics
    with gr.Accordion("Model Info", open=False):
        gr.Markdown(model_info_text())

    inputs = [age, income, debt, years_employed, number_of_loans,
              credit_utilization, late_payments_12m,
              home_ownership, education, payment_history, has_mortgage, region]
    predict_btn.click(run_predict, inputs=inputs, outputs=[pred_md, score_plot, prob_plot, json_out])

if __name__ == "__main__":
    print("Model loaded. Launching Gradio on http://127.0.0.1:7860 ...")
    demo.launch(
        server_name="127.0.0.1",   # use "0.0.0.0" if you need external access
        server_port=7860,          # change if port is busy
        inbrowser=True,            # auto-open browser
        share=False,
        show_error=True
    )
# ---------- end app.py ----------
