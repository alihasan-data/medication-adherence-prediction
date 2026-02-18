# medication-adherence-prediction
Medication Adherence Prediction (Machine Learning Project)

This project builds a machine learning model to predict patient medication adherence using synthetic but realistic healthcare data.

It demonstrates:

Clinical domain insight (PharmD background)

End-to-end data science workflow

Feature engineering

Model development using Logistic Regression + Random Forest

Evaluation (ROC/AUC, confusion matrix, precision/recall)

Clean, reusable ML pipeline with preprocessing

This is a strong portfolio project demonstrating how clinical expertise can be augmented with AI/ML to support medication adherence interventions.

This project uses fully synthetic, simulated healthcare data created for educational purposes only. No real patient data, employer data, or protected health information (PHI) was used.

📁 Project Structure
medication-adherence-prediction/
│
├── data/
│   └── med_adherence_synthetic.csv
│
├── models/
│   ├── logreg_adherence.pkl
│   └── rf_adherence.pkl
│
├── notebooks/
│   └── 01_med_adherence_model.ipynb   # Main modeling notebook
│
├── src/
│   └── generate_synthetic_adherence_data.py
│
├── README.md
└── .gitignore


🎯 Project Goal

Medication adherence directly affects:

Outcomes

Hospitalizations

Overall healthcare costs

This project predicts whether a patient will be:

Adherent (1)

Non-adherent (0)

Based on features such as:

Age

Number of medications

Refill gaps

Prior adherence percentage

Chronic conditions

Mental health flag

Copay tier

Plan type

🧪 Dataset

A synthetic but realistic dataset of 4,000 patients generated using:

src/generate_synthetic_adherence_data.py


Variables include:

| Feature              | Description                         |
| -------------------- | ----------------------------------- |
| age                  | Patient age                         |
| gender               | M/F                                 |
| chronic_conditions   | Count of chronic diseases           |
| num_meds             | Number of medications               |
| refill_gap_days      | Days without medication supply      |
| prior_year_adherence | Percent adherence last year         |
| mental_health_flag   | Depression/anxiety (0/1)            |
| copay_tier           | low / medium / high                 |
| plan_type            | Commercial / Medicare / Medicaid    |
| adherent             | Target (1=adherent, 0=non-adherent) |

🤖 Models Trained
1. Logistic Regression

AUC: 0.769

Good linear baseline

Interpretable for clinicians

2. Random Forest

AUC: 0.757

Better at capturing nonlinear patterns

Handles interactions automatically

Both models use a Pipeline with:

OneHotEncoding for categoricals

Passthrough for numerical features

Clean, end-to-end workflow

📈 Evaluation
| Model               | AUC       |
| ------------------- | --------- |
| Logistic Regression | **0.769** |
| Random Forest       | **0.757** |


ROC Curve

Your notebook includes a combined ROC curve comparing both models.

Confusion Matrices

Both models show balanced performance across classes, indicating good predictive signal without overfitting.

🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-Learn

Matplotlib / Seaborn

Jupyter Notebook

Joblib for model saving

🚀 How to Run
1️⃣ Generate the dataset:
python src/generate_synthetic_adherence_data.py

2️⃣ Open the modeling notebook:
notebooks/01_med_adherence_model.ipynb

3️⃣ Run all cells to train + evaluate models.

## 🧠 Model Artifacts

This project includes one saved model:

logreg_adherence.pkl — Logistic Regression model (small file, included in the repo)

The Random Forest model file was not included due to GitHub’s 100 MB file limit:

rf_adherence.pkl — excluded (too large for GitHub)

To recreate both models locally:

Generate the dataset (if not already created):

python src/generate_synthetic_adherence_data.py


Open the notebook:

notebooks/01_med_adherence_model.ipynb


Run all cells to retrain Logistic Regression and Random Forest.

The notebook will save both models to:

/models/logreg_adherence.pkl
/models/rf_adherence.pkl

🔮 Future Enhancements

Add XGBoost / LightGBM

SHAP values for explainability

Build a Streamlit web app to score new patients

Create a feature importance dashboard

Compare with logistic baseline + boosted models

👤 Author

Ali Hasan, PharmD
Applying clinical expertise to AI-driven healthcare analytics.
