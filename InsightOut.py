
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from pathlib import Path

st.set_page_config(
    page_title="InsightOut: Mental Health Dashboard",
    layout="wide"
)

INSIGHTOUT_STRESS_MAP = {
    'Low':    '#4C78A8',   
    'Medium': '#F58518',   
    'High':   '#E45756'    
}


pio.templates["insightout"] = pio.templates["plotly_white"]
pio.templates["insightout"].layout.update(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=14, color="#2C3E50"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    colorway=['#00A3A3', '#1F3A5F', '#FFB703', '#4C78A8', '#F58518', '#E45756'],
    margin=dict(l=40, r=20, t=60, b=40)
)
pio.templates.default = "insightout"


st.markdown("""
<style>
/* Page & containers */
.stApp, .main { background: #E6F4F1; }
.block-container { padding-top: 1.2rem; }

/* Headings */
h1, h2, h3 { color: #1F3A5F; letter-spacing: .2px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DDE8E6; }

/* Dataframe header */
[data-testid="stDataFrame"] thead tr th {
  background: #F4FBFA !important; color: #1F3A5F !important; font-weight: 600;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 8px; box-shadow: 0 1px 0 rgba(0,0,0,.03);
}

/* Progress bar gradient */
div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #00A3A3, #1F3A5F);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=True)
def load_data():
    candidates = [
        Path("mental_health_data.csv"),
        Path("data/mental_health_data.csv"),
        Path("datasets/mental_health_data.csv")
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Dataset not found. Place 'mental_health_data.csv' next to this script "
            "or in a 'data/' folder."
        )

    data = pd.read_csv(path)


    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace('Prefer not to say', 'Other')
    if 'Stress_Level' in data.columns:
        data['Stress_Level'] = data['Stress_Level'].fillna('Medium')


    if 'Stress_Level' in data.columns:
        stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
        data['Stress_Numeric'] = data['Stress_Level'].map(stress_map)


    numeric_cols = [c for c in ['Sleep_Hours','Work_Hours','Physical_Activity_Hours','Social_Media_Usage'] if c in data.columns]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    if numeric_cols:
        data.dropna(subset=numeric_cols, inplace=True)

    return data

try:
    data = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to:", [
    "Home",
    "Dashboard",
    "Insights & Recommendations",
    "Risk Predictor"
])


if page == "Home":
    st.title("Mental Health & Lifestyle Dashboard")
    st.caption("A data-driven initiative to raise awareness of lifestyle impact on mental well-being.")

    st.subheader("Team Members")
    team_data = {
        "Name": [
            "Muhammad Rifqi Rusydi Bin Rashidi Himzal",
            "Muhammad Armand Faris Bin Md. Rosdi",
            "Ahmad Lutfi Bin Ahmad Shaarizan",
            "Adam Zikry Bin Lizam",
            "Muhammad Syahmi Nuaim bin Anam"
        ],
        "ID": ["22009429", "22009190", "22009343", "21000893", "21000530"],
        "Email": [
            "muhammad_22009429@utp.edu.my",
            "muhammad_22009190@utp.edu.my",
            "ahmad_22009343@utp.edu.my",
            "adam_21000893@utp.edu.my",
            "muhammad_21000530@utp.edu.my"
        ]
    }
    team_df = pd.DataFrame(team_data)
    st.dataframe(team_df, width="stretch", hide_index=True)

    st.markdown("""---""")

    st.subheader("Project Overview")
    st.markdown("""
*Group Name:* InsightOut  
*Topic:* Impact of Lifestyle Choices on Mental Health  
*Objective / Function:* Analyze and visualize how lifestyle factors such as sleep, diet, work hours, and physical activity affect mental well-being.  
*Dataset:* [Zenodo: Mental Health and Lifestyle Dataset](https://zenodo.org/records/14838680)
""")

# ------------------ DASHBOARD ------------------
elif page == "Dashboard":
    st.subheader("Explore Lifestyle & Stress Patterns")

    # Filters
    cols = st.columns(3)
    occs = ['All'] + sorted(data['Occupation'].dropna().unique()) if 'Occupation' in data.columns else ['All']
    countries = ['All'] + sorted(data['Country'].dropna().unique()) if 'Country' in data.columns else ['All']
    genders = ['All'] + sorted(data['Gender'].dropna().unique()) if 'Gender' in data.columns else ['All']

    occupation_filter = cols[0].selectbox("Filter by Occupation", occs)
    country_filter = cols[1].selectbox("Filter by Country", countries)
    gender_filter = cols[2].selectbox("Filter by Gender", genders)

    filtered = data.copy()
    if 'Occupation' in filtered.columns and occupation_filter != 'All':
        filtered = filtered[filtered['Occupation'] == occupation_filter]
    if 'Country' in filtered.columns and country_filter != 'All':
        filtered = filtered[filtered['Country'] == country_filter]
    if 'Gender' in filtered.columns and gender_filter != 'All':
        filtered = filtered[filtered['Gender'] == gender_filter]

    if filtered.empty:
        st.warning("No data matches your selected filters.")
        st.stop()

    # Sidebar chart selector
    st.sidebar.header("Additional Chart Preview")
    extra_chart = st.sidebar.selectbox("Select Additional Chart", [
        "Occupation vs Stress Level",
        "Correlation / Pairwise",
        "Stress Level Distribution",
        "Diet vs Mental Health",
        "Physical Activity by Occupation"
    ])

    # Charts
    if extra_chart == "Stress Level Distribution" and 'Stress_Level' in filtered.columns:
        chart_type = st.sidebar.selectbox("Chart Type", ["Histogram", "Pie"])
        if chart_type == "Histogram":
            fig = px.histogram(
                filtered, x="Stress_Level", color="Stress_Level",
                color_discrete_map=INSIGHTOUT_STRESS_MAP,
                title="Stress Level Histogram"
            )
        else:
            fig = px.pie(
                filtered, names="Stress_Level", color="Stress_Level",
                color_discrete_map=INSIGHTOUT_STRESS_MAP,
                title="Stress Level Pie Chart"
            )
        st.plotly_chart(fig, width="stretch")

    elif extra_chart == "Occupation vs Stress Level" and {'Occupation','Stress_Level'}.issubset(filtered.columns):
        chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Pie"])
        if chart_type == "Bar":
            fig = px.histogram(
                filtered, x="Occupation", color="Stress_Level", barmode="group",
                color_discrete_map=INSIGHTOUT_STRESS_MAP,
                title="Occupation vs Stress Level"
            )
        else:
            # Pie by occupation count (stress color applied to show legend)
            fig = px.pie(
                filtered, names="Occupation", title="Occupation Distribution",
            )
        st.plotly_chart(fig, width="stretch")

    elif extra_chart == "Diet vs Mental Health" and {'Diet_Quality','Mental_Health_Condition'}.issubset(filtered.columns):
        chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Pie"])
        if chart_type == "Bar":
            fig = px.bar(
                filtered, x="Diet_Quality", color="Mental_Health_Condition",
                barmode="group", title="Diet Quality vs Mental Health"
            )
        else:
            fig = px.pie(
                filtered, names="Diet_Quality", color="Mental_Health_Condition",
                title="Diet vs Mental Health Pie"
            )
        st.plotly_chart(fig, width="stretch")

    elif extra_chart == "Physical Activity by Occupation" and {'Occupation','Physical_Activity_Hours','Stress_Level'}.issubset(filtered.columns):
        chart_type = st.sidebar.selectbox("Chart Type", ["Box", "Violin"])
        if chart_type == "Box":
            fig = px.box(
                filtered, x="Occupation", y="Physical_Activity_Hours", color="Stress_Level",
                color_discrete_map=INSIGHTOUT_STRESS_MAP,
                title="Physical Activity by Occupation (Box)"
            )
        else:
            fig = px.violin(
                filtered, x="Occupation", y="Physical_Activity_Hours", color="Stress_Level",
                box=True, points="all",
                color_discrete_map=INSIGHTOUT_STRESS_MAP,
                title="Physical Activity by Occupation (Violin)"
            )
        st.plotly_chart(fig, width="stretch")

    elif extra_chart == "Correlation / Pairwise":
        chart_type = st.sidebar.selectbox("Chart Type", ["Heatmap", "Scatter Matrix"])
        numeric_cols = [c for c in ["Sleep_Hours","Work_Hours","Social_Media_Usage","Physical_Activity_Hours"] if c in filtered.columns]
        if chart_type == "Heatmap" and len(numeric_cols) >= 2:
            fig = px.imshow(
                filtered[numeric_cols].corr(),
                text_auto=True,
                color_continuous_scale=["#E6F4F1", "#00A3A3", "#1F3A5F"],
                title="Correlation Heatmap"
            )
        else:
            base_dims = [c for c in ["Sleep_Hours","Work_Hours","Social_Media_Usage","Physical_Activity_Hours"] if c in filtered.columns]
            color_col = "Stress_Level" if "Stress_Level" in filtered.columns else None
            fig = px.scatter_matrix(
                filtered, dimensions=base_dims,
                color=color_col,
                color_discrete_map=INSIGHTOUT_STRESS_MAP if color_col else None,
                title="Pairwise Relationships"
            )
        st.plotly_chart(fig, width="stretch")


elif page == "Insights & Recommendations":
    st.subheader("Data-Driven Insights")
    st.info("""
*Based on patterns observed in the dataset, several lifestyle behaviors show notable impact on stress and mental well-being:*
""")
    st.markdown("""
- *Sleep Duration:* <6 hours/day → higher stress/anxiety.  
  > Optimal range: *7–8 hours/day*
- *Workload:* >60 hours/week → elevated stress/burnout.  
  > Sustainable: *40–50 hours/week*
- *Diet Quality:* Balanced diets correlate with lower mental health risk.  
  > Focus: fruits, whole grains, omega-3 fats
- *Smoking:* Frequent smoking ↗ stress & mood issues.  
  > Aim for reduction/cessation support
- *Physical Activity:* *7+ hrs/week* → lowest stress levels.  
  > Suggested: *3–5 hrs/week*
- *Social Media:* >3 hrs/day → distraction & mood dips.  
  > Target: *<2–3 hrs/day*
""")

    st.subheader("Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.warning("""
### 🧍 For Individuals
- Sleep *7–8 hours* nightly  
- Exercise *3–7 hours/week*  
- Balanced diet, whole foods  
- Social media *<3 hours/day*  
- Stress-relief routines  
- Seek support early
""")
    with c2:
        st.info("""
### For Organizations
- Work-life balance & flexibility  
- Mental health workshops  
- Peer support / check-ins  
- Encourage breaks & leave  
- Access to counseling/EAP
""")

    st.subheader("📈 Expected Positive Outcomes")
    st.success("""
- Reduce *high-stress cases* by ~*20–30%*  
- Improve overall mental health & satisfaction  
- Boost productivity & creativity  
- Strengthen culture & retention
""")
    st.caption("Correlations ≠ causation. For health decisions, consult professionals.")


elif page == "Risk Predictor":
    st.subheader("Personalized Risk Prediction")

    st.sidebar.markdown("### Input Your Lifestyle Data")
    st.sidebar.header("Lifestyle Inputs")
    st.sidebar.markdown("Adjust sliders to personalize your *Risk Prediction*.")


    input_sleep = st.sidebar.slider("Sleep Hours per Day", 4.0, 10.0, 7.0, 0.1)
    input_work = st.sidebar.slider("Work Hours per Week", 30, 80, 45, 1)
    input_activity = st.sidebar.slider("Physical Activity (hrs/week)", 0, 10, 3, 1)
    input_social = st.sidebar.slider("Social Media (hrs/day)", 0.0, 6.0, 2.5, 0.1)

    gender_classes = sorted(data['Gender'].dropna().unique()) if 'Gender' in data.columns else ["Other"]
    diet_classes = sorted(data['Diet_Quality'].dropna().unique()) if 'Diet_Quality' in data.columns else ["Average"]
    stress_classes = ['Low', 'Medium', 'High']

    input_gender_str = st.sidebar.selectbox("Gender", gender_classes)
    input_diet_str = st.sidebar.selectbox("Diet Quality", diet_classes)
    input_stress_str = st.sidebar.selectbox("Stress Level", stress_classes)

    try:
        needed_cols = ['Sleep_Hours','Work_Hours','Physical_Activity_Hours','Social_Media_Usage',
                       'Gender','Diet_Quality','Stress_Level','Mental_Health_Condition']
        for c in needed_cols:
            if c not in data.columns:
                raise ValueError(f"Missing column in dataset: {c}")

        encoders = {col: LabelEncoder().fit(data[col]) for col in
                    ['Gender','Diet_Quality','Stress_Level','Mental_Health_Condition']}

        model_data = data.copy()
        for col, le in encoders.items():
            model_data[col + '_Enc'] = le.transform(model_data[col])

        X = model_data[['Sleep_Hours','Work_Hours','Physical_Activity_Hours',
                        'Social_Media_Usage','Gender_Enc','Diet_Quality_Enc','Stress_Level_Enc']]
        y = model_data['Mental_Health_Condition_Enc']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
            max_depth=8,
            min_samples_split=5
        )
        rf.fit(X_train, y_train)

        calibrated = CalibratedClassifierCV(estimator=RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
            max_depth=8,
            min_samples_split=5
        ), cv=5)
        calibrated.fit(X_train, y_train)


        y_pred = calibrated.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.info(f"*Model Accuracy:* {accuracy*100:.2f}% (Random Forest + SMOTE + Calibration)")

        user_vector = [[
            input_sleep,
            input_work,
            input_activity,
            input_social,
            encoders['Gender'].transform([input_gender_str])[0],
            encoders['Diet_Quality'].transform([input_diet_str])[0],
            encoders['Stress_Level'].transform([input_stress_str])[0]
        ]]
        user_vector_scaled = scaler.transform(user_vector)

        probs = calibrated.predict_proba(user_vector_scaled)[0]
        yes_index = list(encoders['Mental_Health_Condition'].classes_).index('Yes')
        risk_scaled = float(np.clip(probs[yes_index] * 100, 0, 100))

        st.markdown("---")
        st.subheader("Your Risk Assessment")

        if risk_scaled < 33:
            level = 'Low Risk'
            tier_advice = "Maintain healthy habits and monitor stress regularly."
        elif risk_scaled < 66:
            level = 'Moderate Risk'
            tier_advice = "Consider adjusting lifestyle factors and stress management."
        else:
            level = 'High Risk'
            tier_advice = "Immediate lifestyle changes advised; consult a professional if needed."

        st.metric(label=f"{level}", value=f"{risk_scaled:.2f}%")
        st.progress(int(risk_scaled))

        st.markdown("Personalized Suggestions:")
        suggestions = []
        if input_sleep < 6: suggestions.append("- Increase sleep to 7–9 hours/day. Sleep deficiency raises risk.")
        if input_work > 50: suggestions.append("- Reduce excessive work or schedule breaks to curb stress.")
        if input_activity < 3: suggestions.append("- Add more physical activity; exercise improves mood & resilience.")
        if input_social > 3: suggestions.append("- Limit social media; high screen time can worsen anxiety/stress.")
        if input_stress_str == "High": suggestions.append("- Practice stress reduction: meditation, deep breathing, yoga.")
        if input_diet_str in ["Poor", "Average"]: suggestions.append("- Improve diet quality with whole, nutrient-dense foods.")

        if suggestions:
            for s in suggestions: st.write(s)
        else:
            st.write("- Your lifestyle inputs are balanced. Keep it up! ")

        st.markdown("Top Factors Influencing Risk (from Model):")
        importances = rf.feature_importances_
        feature_names = ['Sleep_Hours','Work_Hours','Physical_Activity_Hours',
                         'Social_Media_Usage','Gender (enc)','Diet_Quality (enc)','Stress_Level (enc)']
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx[:3]:
            st.write(f"*{feature_names[i]}* — importance: {importances[i]:.3f}")

        st.markdown("Lifestyle Comparison:")
        avg_data = data[['Sleep_Hours','Work_Hours','Physical_Activity_Hours','Social_Media_Usage']].mean()
        your_vals = [input_sleep, input_work, input_activity, input_social]
        comp_df = pd.DataFrame({
            "Metric": ['Sleep_Hours','Work_Hours','Physical_Activity_Hours','Social_Media_Usage'],
            "Your Input": your_vals,
            "Dataset Average": [avg_data.get(k, np.nan) for k in ['Sleep_Hours','Work_Hours','Physical_Activity_Hours','Social_Media_Usage']]
        })
        fig_comp = px.bar(
            comp_df.melt(id_vars="Metric", var_name="Series", value_name="Value"),
            x="Metric", y="Value", color="Series",
            barmode="group", title="Your Inputs vs Dataset Average"
        )
        st.plotly_chart(fig_comp, width="stretch")

        st.markdown("Suggested Next Steps:")
        st.write(f"- {tier_advice}")
        if risk_scaled > 66:
            st.write("- Consider speaking with a mental health professional.")
            st.write("- Review lifestyle factors: sleep, diet, exercise, social media, stress management.")

    except Exception as e:
        st.error(f"Model error: {e}")