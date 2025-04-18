import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load saved models & preprocessors
model = joblib.load("career_model.pkl")
scaler = joblib.load("scaler.pkl")
vt = joblib.load("vt.pkl")
skb = joblib.load("skb.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset
df = pd.read_csv("Final_Modified_Project.csv")

# Handle missing values
for col in df.select_dtypes(include='object').columns:
    if col != 'Career':
        df[col] = df[col].fillna(df[col].mode()[0])
        le = label_encoders.get(col)
        if le:
            df[col] = le.transform(df[col])


# Field Options & Skills
field_options = [
    "Engineering", "Medical", "Business", "Arts", "Law", "Science"
]

skills_options = [
    "Programming, Data Structures, Algorithms",
    "Structural Analysis, Construction Management",
    "Thermodynamics, CAD Design, Manufacturing",
    "Anatomy, Physiology, Clinical Diagnosis",
    "Drug Formulation, Pharmaceutical Chemistry",
    "Accounting, Financial Modelling, Risk Management",
    "Market Research, Brand Management, Advertising",
    "UI/UX Design, Illustration, Typography",
    "Creative Writing, Editing, Content Strategy",
    "Legal Research, Criminal Law, Civil Law",
    "Machine Learning, Statistics, Data Visualization",
    "Genetics, Microbiology, Bioinformatics"
]

# Field-specific relevant skills
field_skill_map = {
    "Engineering": ["Aptitude_Score", "Technical_Knowledge", "Problem_Solving", "Teamwork",
                    "Certifications_Completed", "Internships_Completed", "Experience_Years",
                    "Projects_Done", "Industry_Exposure"],
    "Medical": ["Aptitude_Score", "Problem_Solving", "Teamwork", "Communication_Skills",
                "Internships_Completed", "Experience_Years", "Industry_Exposure"],
    "Law": ["Aptitude_Score", "Leadership", "Communication_Skills", "Problem_Solving",
            "Experience_Years", "Teamwork", "Industry_Exposure"],
    "Arts": ["Creativity", "Communication_Skills", "Projects_Done", "Experience_Years"],
    "Business": ["Communication_Skills", "Leadership", "Problem_Solving", "Experience_Years",
                 "Certifications_Completed", "Projects_Done", "Industry_Exposure"],
    "Science": ["Aptitude_Score", "Technical_Knowledge", "Problem_Solving", "Certifications_Completed",
                "Experience_Years", "Projects_Done"]
}

# UI setup
st.set_page_config(page_title="Career Recommendation System", layout="wide")
st.title("🔮 Career Recommendation System")
st.write("This system predicts your ideal career based on your skills, experience, and interests.")

# Input function
def get_user_input():
    st.sidebar.header("Enter Your Details")

    field = st.sidebar.selectbox("Field", field_options)
    skills = st.sidebar.selectbox("Specialization Skills", skills_options)
    aptitude_score = st.sidebar.slider("Aptitude Score (1-100)", 1, 100, 60)
    technical_knowledge = st.sidebar.slider("Technical Knowledge (1-100)", 1, 100, 70)
    problem_solving = st.sidebar.slider("Problem Solving (1-100)", 1, 100, 65)
    communication = st.sidebar.slider("Communication Skills (1-100)", 1, 100, 75)
    creativity = st.sidebar.slider("Creativity (1-100)", 1, 100, 60)
    leadership = st.sidebar.slider("Leadership (1-100)", 1, 100, 55)
    teamwork = st.sidebar.slider("Teamwork (1-100)", 1, 100, 70)
    certifications = st.sidebar.slider("Certifications Completed", 0, 10, 2)
    internships = st.sidebar.slider("Internships Completed", 0, 10, 3)
    experience_years = st.sidebar.slider("Experience Years", 0, 20, 5)
    projects_done = st.sidebar.slider("Projects Done", 0, 15, 4)
    industry_exposure = st.sidebar.slider("Industry Exposure (1-10)", 1, 10, 6)

    def safe_encode(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else -1

    field_encoded = safe_encode(label_encoders['Field'], field)
    skills_encoded = safe_encode(label_encoders['Specialization_Skills'], skills)

    user_data = np.array([[field_encoded, skills_encoded, aptitude_score, technical_knowledge, problem_solving,
                           communication, creativity, leadership, teamwork, certifications, internships,
                           experience_years, projects_done, industry_exposure]])

    return user_data, field, field_encoded, skills

# Collect user input
input_data, selected_field, encoded_field, selected_skills = get_user_input()

# Prediction
input_transformed = vt.transform(input_data)
input_transformed = skb.transform(input_transformed)
input_transformed = scaler.transform(input_transformed)
prediction = model.predict(input_transformed)[0]
predicted_career = label_encoders['Career'].inverse_transform([prediction])[0]

career_alignment = {
    "Engineering": ["Software Engineer", "Mechanical Engineer", "Civil Engineer"],
    "Medical": ["Doctor", "Pharmacist"],
    "Business": ["Marketing Manager", "Financial Analyst"],
    "Arts": ["Graphic Designer", "Writer"],
    "Law": ["Lawyer"],
    "Science": ["Data Scientist", "Biotechnologist"]
}

# Show result
if predicted_career not in career_alignment.get(selected_field, []):
    st.warning(f"⚠️ Your chosen field **({selected_field})** and entered skills do not align!")
else:
    st.subheader(f"✅ Recommended Career: **{predicted_career}**")

    # 🧠 KMeans Clustering (only if career aligned)
    career_skill_map = {
        "Doctor": ["Teamwork", "Communication_Skills", "Internships_Completed", "Experience_Years"],
        "Pharmacist": ["Communication_Skills", "Certifications_Completed", "Industry_Exposure"],
        "Software Engineer": ["Aptitude_Score","Technical_Knowledge", "Problem_Solving", "Projects_Done", "Certifications_Completed"],
        "Civil Engineer": ["Aptitude_Score","Problem_Solving", "Teamwork", "Experience_Years", "Projects_Done"],
        "Marketing Manager": ["Communication_Skills", "Leadership", "Projects_Done", "Industry_Exposure"],
        "Writer": ["Creativity", "Communication_Skills", "Projects_Done"],
        "Lawyer": ["Communication_Skills", "Problem_Solving", "Leadership", "Experience_Years"],
        "Data Scientist": ["Aptitude_Score","Technical_Knowledge", "Problem_Solving", "Certifications_Completed", "Projects_Done"],
        "Biotechnologist": ["Experience_Years", "Internships_Completed", "Industry_Exposure"]
    }

    selected_skill_features = career_skill_map.get(predicted_career, field_skill_map.get(selected_field, []))
    cluster_features = df.drop(columns=['Career'])
    field_data = cluster_features[cluster_features['Field'] == encoded_field].copy()

    if len(field_data) >= 4 and all(feat in field_data.columns for feat in selected_skill_features):
        kmeans_data = field_data[selected_skill_features].copy()

        # Fit clustering model
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        field_data["Cluster"] = kmeans.fit_predict(kmeans_data)

        cluster_means = kmeans_data.copy()
        cluster_means["Cluster"] = field_data["Cluster"]
        cluster_avg = cluster_means.groupby("Cluster").mean().mean(axis=1).sort_values()
        level_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
        level_map = {cluster: level_names[i] for i, cluster in enumerate(cluster_avg.index)}
        field_data["Level"] = field_data["Cluster"].map(level_map)

        user_vector = input_data[0]
        user_skill_input = user_vector[2:]  # Strip field and skills
        user_dict = dict(zip([
            "Aptitude_Score", "Technical_Knowledge", "Problem_Solving", "Communication_Skills",
            "Creativity", "Leadership", "Teamwork", "Certifications_Completed",
            "Internships_Completed", "Experience_Years", "Projects_Done", "Industry_Exposure"
        ], user_skill_input))

        user_selected_vector = np.array([user_dict[feat] for feat in selected_skill_features])
        cluster_label = kmeans.predict([user_selected_vector])[0]
        user_level = level_map.get(cluster_label, "Unknown")

        st.info(f"🎓 Based on your skills, your **expertise level** in **{selected_field}** is: **{user_level}**")

        expert_cluster_id = None
        for cid, lvl in level_map.items():
            if lvl == "Expert":
                expert_cluster_id = cid
                break

        if expert_cluster_id is not None:
            expert_centroid = kmeans.cluster_centers_[expert_cluster_id]
            diff = expert_centroid - user_selected_vector

            lacking_skills = [selected_skill_features[i] for i, d in enumerate(diff) if d > 10]

            if lacking_skills:
                st.subheader("🛠️ Skills to Improve to Become Expert:")
                for skill in lacking_skills:
                    st.markdown(f"- **{skill.replace('_', ' ')}**")
            else:
                st.success("🎉 You are already close to Expert level in your field!")
        else:
            st.warning("⚠️ Expert cluster not found in this field.")
    else:
        st.warning("⚠️ Not enough data or matching features for clustering in this field.")
