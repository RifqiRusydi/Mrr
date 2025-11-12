# streamlit_mental_health_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import io
import os

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Mental Health & Lifestyle Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Mental Health & Lifestyle Analyzer")
st.markdown("""
Analyze how lifestyle factors such as **sleep, exercise, diet, and work hours** affect **mental health conditions**.  
This system automatically loads your dataset, visualizes insights, predicts risks, and gives personalized recommendations.
""")

# =========================
# LOAD LOCAL DATASET
# =========================
# 👉 Change the path below to your actual CSV file location
DATA_PATH = r"mental_health_data.csv"   # <-- your dataset filename or full path

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success(f"✅ Dataset loaded successfully from: {DATA_PATH}")
else:
    st.error(f"❌ Dataset not found at: {DATA_PATH}")
    st.stop()

# =========================
# DATA PREVIEW
# =========================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
st.write("**Shape:**", df.shape)
st.write("**Columns:**", list(df.columns))

# =========================
# MAIN TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dashboard",
    "🤖 Prediction Tool",
    "💡 Recommendations",
    "📄 Download Report"
])

# =========================
# DASHBOARD TAB
# =========================
with tab1:
    st.header("📊 Dashboard: Insights and Analysis")

    target_col = st.selectbox("Select Target Column (e.g., Stress_Level, Mental_Health_Status)", df.columns)

    # Distribution
    st.write("### 🔹 Target Variable Distribution")
    fig, ax = plt.subplots()
    df[target_col].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {target_col}")
    st.pyplot(fig)

    # Correlation heatmap
    st.write("### 🔹 Correlation Heatmap (Numerical Features)")
    num_df = df.select_dtypes(include=['number'])
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation.")

    # Summary
    st.write("### 🔹 Summary Statistics")
    st.dataframe(df.describe())

# =========================
# PREDICTION TAB
# =========================
with tab2:
    st.header("🤖 Mental Health Risk Prediction")

    feature_cols = st.multiselect(
        "Select Feature Columns (sleep, exercise, diet, work hours, etc.)", 
        df.columns
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    target_col = st.selectbox("Select Target Column", df.columns)

    X = df[feature_cols]
    y = df[target_col]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    y = pd.factorize(y)[0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    # User input form
    st.subheader("🔹 Enter Your Lifestyle Data for Prediction")
    user_data = {}
    for col in X.columns:
        default_val = float(df[col].mean()) if np.issubdtype(df[col].dtype, np.number) else 0.0
        user_data[col] = st.number_input(f"{col}", value=default_val)

    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)[0]
    st.success(f"🧾 Predicted Mental Health Category: **{prediction}**")

# =========================
# RECOMMENDATIONS TAB
# =========================
with tab3:
    st.header("💡 Personalized Lifestyle Recommendations")

    st.markdown("""
    Based on your selected features and input, here are tailored tips for improving mental well-being:
    """)

    recs = []
    col_lower = [c.lower() for c in feature_cols]

    if any("sleep" in c for c in col_lower):
        recs.append("🛌 Maintain a consistent sleep schedule of 7–9 hours per night.")
    if any("diet" in c for c in col_lower):
        recs.append("🥗 Eat balanced meals with sufficient vitamins and hydration.")
    if any("exercise" in c for c in col_lower):
        recs.append("🏃‍♀️ Engage in regular physical activity (30 mins/day).")
    if any("work" in c or "hours" in c for c in col_lower):
        recs.append("🧘 Take regular breaks to manage work-life balance.")
    if len(recs) == 0:
        recs.append("💡 Add lifestyle-related features (sleep, diet, exercise, work) for more specific recommendations.")

    for r in recs:
        st.write(r)

# =========================
# REPORT DOWNLOAD TAB
# =========================
with tab4:
    st.header("📄 Generate Downloadable Report")

    st.markdown("You can generate a summary report containing prediction and recommendations as a PDF file.")

    if st.button("🖨️ Generate PDF Report"):
        buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Mental Health & Lifestyle Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"Model Accuracy: {acc*100:.2f}%")
        pdf.multi_cell(0, 10, f"Predicted Mental Health Category: {prediction}")
        pdf.ln(5)
        pdf.cell(0, 10, "Recommendations:", ln=True)
        for r in recs:
            pdf.multi_cell(0, 8, f"- {r}")
        pdf.output(buffer)
        st.download_button(
            label="📥 Download PDF",
            data=buffer.getvalue(),
            file_name="mental_health_report.pdf",
            mime="application/pdf"
        )

st.success("✅ Application Ready — All systems loaded automatically.")