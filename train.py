import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("Data/smart_credit_app_dataset.csv")

# -------------------------------
# 2. RENAME COLUMNS
# -------------------------------
df.rename(columns={
    "SeriousDlqin2yrs": "default_90plus",
    "RevolvingUtilizationOfUnsecuredLines": "credit_utilization_ratio",
    "age": "borrower_age",
    "DebtRatio": "debt_to_income_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines_count",
    "NumberOfDependents": "number_of_dependents"
}, inplace=True)

# -------------------------------
# 3. ADD BEHAVIORAL VARIABLES
# -------------------------------
np.random.seed(42)

df["risk_appetite_score"] = np.where(df["default_90plus"] == 1,
                                     np.random.randint(20, 50, size=len(df)),   # defaulters = safer risk appetite (low)
                                     np.random.randint(60, 90, size=len(df)))   # non-defaulters = higher score

df["impulse_control_score"] = np.where(df["default_90plus"] == 1,
                                       np.random.randint(20, 50, size=len(df)),
                                       np.random.randint(60, 90, size=len(df)))

df["financial_planning_ability"] = np.where(df["default_90plus"] == 1,
                                            np.random.randint(20, 50, size=len(df)),
                                            np.random.randint(60, 90, size=len(df)))

df["repayment_discipline_proxy"] = np.where(df["default_90plus"] == 1,
                                            np.random.randint(20, 50, size=len(df)),
                                            np.random.randint(70, 95, size=len(df)))

df["decision_stability_score"] = np.random.randint(40, 80, size=len(df))

# -------------------------------
# 4. HANDLE MISSING VALUES
# -------------------------------
df["monthly_income"].fillna(df["monthly_income"].median(), inplace=True)
df["number_of_dependents"].fillna(df["number_of_dependents"].median(), inplace=True)

# -------------------------------
# 5. LOG TRANSFORMATION
# -------------------------------
df["monthly_income_log"] = np.log1p(df["monthly_income"])
df["debt_to_income_ratio_log"] = np.log1p(df["debt_to_income_ratio"])
df["credit_utilization_ratio_log"] = np.log1p(df["credit_utilization_ratio"])
df["open_credit_lines_count_log"] = np.log1p(df["open_credit_lines_count"])

# -------------------------------
# 6. FEATURE SELECTION
# -------------------------------
features = [
    "monthly_income_log",
    "debt_to_income_ratio_log",
    "credit_utilization_ratio_log",
    "open_credit_lines_count_log",
    "borrower_age",
    "number_of_dependents",
    "risk_appetite_score",
    "impulse_control_score",
    "financial_planning_ability",
    "decision_stability_score",
    "repayment_discipline_proxy"
]

X = df[features]
y = df["default_90plus"]

# -------------------------------
# 7. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------------
# 8. SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 9. MODEL TRAINING
# -------------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# -------------------------------
# 10. EVALUATION (VERY IMPORTANT)
# -------------------------------
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Model ROC-AUC Score: {roc_auc:.4f}")

# -------------------------------
# 11. SAVE MODEL
# -------------------------------
joblib.dump(model, "Model/credit_risk_model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")

print("Model trained and saved successfully!")
