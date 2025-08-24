# 💳 Credit Scoring Model

**License:** MIT &nbsp;|&nbsp; **Built with:** Gradio UI + scikit-learn (+ XGBoost)  

A machine learning web app that predicts borrower creditworthiness (**Good / Standard / Poor**) from profile and financial inputs.  
The app ships with a polished, high-contrast UI (Gradio), shows a **0–100 risk score**, class probabilities, and helpful extra stats (DTI, utilization, late pays).

---

## 📊 Features
- Predict credit class: **Good / Standard / Poor**
- **Risk Score (0–100)** + risk band (Low/Medium/High)
- Probability bar chart for each class
- Extra stats: **Debt-to-Income**, **Credit Utilization**, **Late Payments**, **#Loans**
- Clean JSON output for easy integration
- Training script with preprocessing (impute/scale/one-hot) and optional **SMOTE**
- Compares **LogReg**, **RandomForest**, **XGBoost** and saves the best pipeline

---

## 🧠 Technologies Used
- Python 3.x
- scikit-learn, imbalanced-learn (SMOTE), XGBoost
- Pandas, NumPy
- Gradio, Matplotlib
- Jupyter Notebook (optional, for exploration)
- VS Code / Colab (either works)

---
## 📁 Folder Structure

credit-scoring-model/
├── app.py # Gradio web app (loads the saved pipeline)
├── train.py # Model training & comparison; saves joblib + metrics
├── credit_scoring_model.joblib # Trained model pipeline
├── metrics.json # Training results (auto-created by train.py)
├── requirements.txt # Python dependencies
├── Output/ # screenshots 
│ ├── predict.jpg # app UI
│ └── result.jpg # example result
├── LICENSE
└── README.md

---

 **🚀 How to Run Locally**
 ```bash

**1) Clone**

git clone https://github.com/YOUR_USERNAME/credit-scoring-model.git
cd credit-scoring-model

**2) Create a virtualenv**

# Windows (PowerShell)
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python3 -m venv .venv
source .venv/bin/activate

**3) Install dependencies**

python -m pip install --upgrade pip
pip install -r requirements.txt

**4) Model Training**
python train.py --data credit_scoring_sample.csv --target Risk_good --smote on
python app.py

**5) Deployment**
python app.py
Open the app at http://127.0.0.1:7860

📜 License
This project is licensed under the MIT License — see LICENSE.

👨‍💻 Author
Adapa Brunda Mani
B.Tech, Artificial Intelligence and Machine Learning
SRM Institute of Science and Technology, Ramapuram
