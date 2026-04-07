import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "Model", "credit_risk_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "Model", "scaler.pkl"))

st.set_page_config(page_title="Smart Credit Risk", layout="centered")

st.title("🧠 Smart Cognitive Credit Risk Assessment")

st.write("Complete the steps below to evaluate your credit risk.")

# =====================================================
# PROGRESS BAR
# =====================================================
progress = st.progress(0)

# =====================================================
# FINANCIAL INPUTS
# =====================================================
st.header("📊 Step 1: Financial Details")

monthly_income = st.number_input("Monthly Income (₹)", 0, 100000, 6000)
debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 2.0, 0.3)
credit_util = st.slider("Credit Utilization Ratio", 0.0, 10.0, 1.0)
open_loans = st.number_input("Open Credit Lines", 0, 10, 2)
age = st.number_input("Age", 18, 70, 25)
dependents = st.number_input("Dependents", 0, 5, 0)

progress.progress(25)

# =====================================================
# BEHAVIORAL GAMES (ADVANCED)
# =====================================================
st.header("🎮 Behavioral Assessment")

# -------------------------------
# 1. RISK APPETITE
# -------------------------------
st.subheader("Game 1: Investment Decision")

risk_q1 = st.radio(
    "You receive ₹10,000 as a bonus. What would you most likely do?",
    [
        "Deposit it in a fixed savings account with guaranteed return",
        "Invest in a mutual fund with moderate risk",
        "Invest in stocks or cryptocurrency with high return potential"
    ]
)

risk_q2 = st.radio(
    "Choose the option you feel more comfortable with:",
    [
        "Guaranteed ₹5,000 gain",
        "50% chance to earn ₹15,000 and 50% chance to earn nothing"
    ]
)

risk_map = {
    "Deposit it in a fixed savings account with guaranteed return": 20,
    "Invest in a mutual fund with moderate risk": 50,
    "Invest in stocks or cryptocurrency with high return potential": 80,
    "Guaranteed ₹5,000 gain": 20,
    "50% chance to earn ₹15,000 and 50% chance to earn nothing": 70
}

risk_score = (risk_map[risk_q1] + risk_map[risk_q2]) / 2

# -------------------------------
# 2. IMPULSE CONTROL
# -------------------------------
st.subheader("Game 2: Spending Behavior")

impulse_q1 = st.radio(
    "You want to buy an expensive gadget. What do you usually do?",
    [
        "Buy immediately if I like it",
        "Think about it for a few days",
        "Compare options and wait for better deals"
    ]
)

impulse_q2 = st.radio(
    "Choose one option:",
    [
        "Receive ₹1,000 today",
        "Receive ₹1,500 after one month"
    ]
)

impulse_map = {
    "Buy immediately if I like it": 20,
    "Think about it for a few days": 50,
    "Compare options and wait for better deals": 80,
    "Receive ₹1,000 today": 20,
    "Receive ₹1,500 after one month": 80
}

impulse_score = (impulse_map[impulse_q1] + impulse_map[impulse_q2]) / 2

# -------------------------------
# 3. FINANCIAL PLANNING
# -------------------------------
st.subheader("Game 3: Budget Planning")

plan_q1 = st.radio(
    "How do you usually allocate your monthly income?",
    [
        "Spend most of it without strict planning",
        "Save around 10–20% regularly",
        "Plan budget carefully and save more than 30%"
    ]
)

plan_q2 = st.radio(
    "If an unexpected ₹5,000 expense occurs, what do you do?",
    [
        "Use credit card or borrow",
        "Use some savings and adjust budget",
        "Pay fully from emergency savings"
    ]
)

plan_map = {
    "Spend most of it without strict planning": 20,
    "Save around 10–20% regularly": 60,
    "Plan budget carefully and save more than 30%": 90,
    "Use credit card or borrow": 30,
    "Use some savings and adjust budget": 60,
    "Pay fully from emergency savings": 90
}

planning_score = (plan_map[plan_q1] + plan_map[plan_q2]) / 2

# -------------------------------
# 4. DECISION CONSISTENCY
# -------------------------------
st.subheader("Game 4: Consistency Check")

stab_q1 = st.radio(
    "You prefer investments that are:",
    ["Safe and predictable", "High risk and high return"]
)

stab_q2 = st.radio(
    "In a similar situation later, you choose:",
    ["Safe and predictable again", "High risk again"]
)

stability_score = 80 if stab_q1.split()[0] == stab_q2.split()[0] else 40

# -------------------------------
# 5. REPAYMENT DISCIPLINE
# -------------------------------
st.subheader("Game 5: Repayment Behavior")

repay_q1 = st.radio(
    "Your loan EMI is due tomorrow. What do you do?",
    [
        "Pay before due date",
        "Pay exactly on due date",
        "Delay payment"
    ]
)

repay_q2 = st.radio(
    "If your income is low in a month:",
    [
        "Still ensure payment by adjusting expenses",
        "Delay or skip payment"
    ]
)

repay_map = {
    "Pay before due date": 90,
    "Pay exactly on due date": 70,
    "Delay payment": 30,
    "Still ensure payment by adjusting expenses": 80,
    "Delay or skip payment": 20
}

repayment_score = (repay_map[repay_q1] + repay_map[repay_q2]) / 2

# -------------------------------
# 6. LOSS AVERSION (NEW)
# -------------------------------
st.subheader("Game 6: Loss Sensitivity")

loss_q = st.radio(
    "Would you accept a gamble where:",
    [
        "You can gain ₹10,000 or lose ₹10,000",
        "You avoid the gamble completely"
    ]
)

loss_score = 70 if loss_q.startswith("You can gain") else 30

# -------------------------------
# 7. TIME PREFERENCE (NEW)
# -------------------------------
st.subheader("Game 7: Future Orientation")

time_q = st.radio(
    "You receive ₹10,000 today. What do you do?",
    [
        "Spend it immediately",
        "Save it for future needs",
        "Invest it for long-term growth"
    ]
)

time_map = {
    "Spend it immediately": 20,
    "Save it for future needs": 60,
    "Invest it for long-term growth": 90
}

time_score = time_map[time_q]

# -------------------------------
# FINAL COMBINED SCORES
# -------------------------------
risk_score = (risk_score + loss_score) / 2
impulse_score = (impulse_score + time_score) / 2
# =====================================================
# SCORE BREAKDOWN
# =====================================================
st.header("📊 Behavioral Score Breakdown")

score_df = pd.DataFrame({
    "Category": ["Risk", "Impulse", "Planning", "Stability", "Repayment"],
    "Score": [risk_score, impulse_score, planning_score, stability_score, repayment_score]
})

st.bar_chart(score_df.set_index("Category"))

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Predict Credit Risk"):

    progress.progress(100)

    df = pd.DataFrame([{
        "monthly_income_log": np.log1p(monthly_income),
        "debt_to_income_ratio_log": np.log1p(debt_ratio),
        "credit_utilization_ratio_log": np.log1p(credit_util),
        "open_credit_lines_count_log": np.log1p(open_loans),
        "borrower_age": age,
        "number_of_dependents": dependents,
        "risk_appetite_score": risk_score,
        "impulse_control_score": impulse_score,
        "financial_planning_ability": planning_score,
        "decision_stability_score": stability_score,
        "repayment_discipline_proxy": repayment_score
    }])

    df_scaled = scaler.transform(df)
    raw_prob = model.predict_proba(df_scaled)[0][1]
    prob = raw_prob * 0.6  # reduce overconfidence
    st.subheader("📈 Final Result")
    st.write(f"Default Probability: {prob*100:.2f}%")

    # Risk Category
    if prob < 0.40:
        st.success("🟢 Low Risk")
    elif prob < 0.70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # =====================================================
    # WHY THIS RESULT (MOST IMPORTANT)
    # =====================================================
    st.header("🧠 Why this result?")

    explanations = []

    if repayment_score < 50:
        explanations.append("Low repayment discipline increases default risk.")
    else:
        explanations.append("Strong repayment behavior reduces risk.")

    if impulse_score < 50:
        explanations.append("Low impulse control indicates higher spending risk.")
    else:
        explanations.append("Good impulse control supports financial stability.")

    if planning_score < 50:
        explanations.append("Weak financial planning increases financial stress.")
    else:
        explanations.append("Strong planning improves repayment ability.")

    if risk_score > 60:
        explanations.append("High risk-taking behavior increases default chances.")
    else:
        explanations.append("Moderate risk-taking behavior is safer.")

    if monthly_income < 5000:
        explanations.append("Lower income increases financial vulnerability.")
    else:
        explanations.append("Stable income supports repayment capacity.")

    for exp in explanations:
        st.write("•", exp)
    st.subheader("📈 Final Result")
    st.write(f"Default Probability: {prob*100:.2f}%")

    # ===============================
    # RISK CATEGORY
    # ===============================
    if prob < 0.40:
        st.success("🟢 Low Risk")
        risk_label = "Low"
    elif prob < 0.70:
        st.warning("🟡 Medium Risk")
        risk_label = "Medium"
    else:
        st.error("🔴 High Risk")
        risk_label = "High"

    # ===============================
    # LOAN ELIGIBILITY SCORE
    # ===============================
    eligibility_score = (1 - prob) * 100
    st.subheader("💰 Loan Eligibility Score")
    st.write(f"You are eligible for loans at approx: **{eligibility_score:.2f}% confidence**")
    # =====================================================
    # SAFE LOAN AMOUNT ESTIMATION
    # =====================================================
    st.subheader("💸 Safe Loan Amount Recommendation")

    # EMI rule (30–40% of income depending on risk)
    if eligibility_score > 75:
        emi_ratio = 0.40
    elif eligibility_score > 50:
        emi_ratio = 0.30
    else:
        emi_ratio = 0.20

    max_emi = monthly_income * emi_ratio

    # Assume 5-year tenure (60 months)
    loan_amount = max_emi * 60

    st.write(f"Based on your profile, a **safe loan amount** would be:")
    st.success(f"₹ {loan_amount:,.0f}")

    # =====================================================
    # INTEREST RATE ESTIMATION
    # =====================================================
    st.subheader("📉 Expected Interest Rate Range")

    if eligibility_score > 75:
        st.success("You can get loans at approx **8% – 11% interest**")

    elif eligibility_score > 50:
        st.warning("Expected interest rate: **11% – 16%**")

    else:
        st.error("Higher risk profile: **16% – 24% interest**")

    # ===============================
    # BANK / LENDER SUGGESTIONS
    # ===============================
    st.subheader("🏦 Recommended Lenders")

    if eligibility_score > 75:
        st.success("You are highly eligible. Recommended banks:")
        st.write("• HDFC Bank")
        st.write("• ICICI Bank")
        st.write("• Axis Bank")
        st.write("• SBI")

    elif eligibility_score > 50:
        st.warning("Moderate eligibility. You may consider:")
        st.write("• NBFCs like Bajaj Finserv")
        st.write("• Tata Capital")
        st.write("• IDFC First Bank")

    else:
        st.error("Low eligibility currently. Consider improving profile before applying.")
        st.write("• Try secured loans (gold loan, FD-backed loan)")
        st.write("• Fintech lenders with flexible criteria")