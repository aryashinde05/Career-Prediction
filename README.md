# 💼 Career Prediction and Skill Gap Analyzer using XGBoost and K-Means

This project presents a **Career Prediction System integrated with a Skill Gap Analyzer**. It uses machine learning models such as **XGBoost Classifier** and **K-Means Clustering** to help users identify their most suitable career path based on their skills, and provides insights on how to bridge the gap to become industry-ready.

---

## 🚀 Features

- 🎯 Predicts career path based on user inputs like skills, certifications, and experience
- 📊 Skill Gap Analyzer highlights missing or weak skills
- 💡 Suggests improvement areas to align with desired careers
- 🔍 Compares new model with existing models for performance evaluation
- 🧠 Trained using **XGBoost Classifier** (achieves ~95% accuracy)
- 🧪 Uses **K-Means Clustering** for skill grouping and analysis

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**:  
  - XGBoost Classifier – for accurate career prediction  
  - K-Means Clustering – for grouping user profiles based on skills
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

---

## 📁 Project Structure
Career-Prediction-SkillGap/
│
├── model/                   # Trained models
├── data/                    # Final_Modified_Project.csv dataset
├── app.py                   # Main Streamlit Application
├── utils.py                 # Supporting functions
├── README.md                # Project Documentation
└── requirements.txt         # Dependencies


Install dependencies
pip install -r requirements.txt
Run the Streamlit app

Run app by
streamlit run app.py
