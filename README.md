# 💰 Individual Income Classification

This project predicts whether a person's annual income is **above $50K or below $50K** based on demographic and personal features such as age, workclass, education, marital status, occupation, relationship, capital gain, capital loss, weekly working hours, and native country.

The dataset used is the **UCI Adult Income Dataset**, containing **48,842 rows** with **14 input features** and **1 target column (Income)**.


## 📌 Workflow
- Split the data into **X_train and X_test first** (to avoid data leakage)
- Performed **EDA only on X_train** — the test data was not touched during analysis
- Removed unwanted columns after EDA
- Applied **OneHotEncoding** to categorical columns
- Applied **StandardScaler** after encoding
- Trained **6 ML models** and compared performance
- **XGBoost Classifier** selected as the final model (**Test Accuracy: 0.8768, F1-score: 0.72** for lowest class)
- Final model integrated in a **Streamlit web application**

## ▶️ Run the Project on Your Computer
**To run this project locally, use the following commands in your terminal:**

pip install -r requirements.txt

streamlit run app.py

**These commands will install the required libraries and open the app in your browser (running locally on your computer).**
