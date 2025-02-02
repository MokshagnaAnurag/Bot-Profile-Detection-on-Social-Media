
# 🤖 Bot Profile Detection on Social Media

### 🚀 **Detecting Automated Bot Profiles on Social Media Platforms**

This project aims to **detect bot accounts** on social media using **machine learning techniques**. By analyzing **user behavior, engagement patterns, and textual content**, our model can classify whether an account is a **bot or a human**.

---

## 📂 **Project Structure**

```
💁️ Bot-Profile-Detection
🗂 data/                # Dataset files (CSV format)
🗂 src/                 # Source code for data processing & model training
    └️ bot_detection.py    # Main ML pipeline (preprocessing, training, testing)
    └️ feature_engineering.py # Feature extraction utilities
🗂 models/              # Saved ML models
💟 requirements.txt     # Required dependencies
💟 README.md            # Project documentation
💟 LICENSE              # Open-source license (MIT)
```

---

## 🛠 **Installation & Setup**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/MokshagnaAnurag/Bot-Profile-Detection-on-Social-Media-.git
cd Bot-Profile-Detection
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Bot Detection Model**

```bash
python src/bot_detection.py
```

---

## 📊 **Dataset Format**

Your dataset should be a **CSV file** with the following columns:

| Column Name      | Description                                       |
| ---------------- | ------------------------------------------------- |
| `User ID`        | Unique identifier for the user                    |
| `Username`       | Username of the account                           |
| `Tweet`          | The tweet content                                 |
| `Created At`     | Timestamp of the tweet                            |
| `Retweet Count`  | Number of retweets                                |
| `Mention Count`  | Number of mentions                                |
| `Follower Count` | Number of followers                               |
| `Verified`       | Whether the account is verified (1 = Yes, 0 = No) |
| `Bot Label`      | Target variable (0 = Human, 1 = Bot)              |

---

## 🎯 **Features Used for Detection**

👉 **Text Analysis:** TF-IDF vectorization of tweets  
👉 **Posting Patterns:** Extraction of posting hour from timestamps  
👉 **Account Statistics:** Retweet count, mentions, follower count, verification status  
👉 **Machine Learning Model:** Logistic Regression  

---

## 📊 **Model Performance**

### **Classification Report:**

| Metric           | Value |
| ----------------- | ----- |
| Accuracy          | 0.5016 |
| Precision (Human) | 0.5012 |
| Precision (Bot)   | 0.5020 |
| Recall (Human)    | 0.5056 |
| Recall (Bot)      | 0.4976 |
| F1-Score (Human)  | 0.5034 |
| F1-Score (Bot)    | 0.4998 |
| AUC-ROC Score     | 0.4997 |

---

## 📚 **Output**

A **CSV report** is generated with bot predictions and confidence scores:

```bash
bot_detection_report.csv
```

| User ID | Username | Tweet            | Predicted Bot | Confidence Score |
| ------- | -------- | ---------------- | ------------- | ---------------- |
| 12345   | @user1   | "Hello World"    | 0 (Human)     | 0.12             |
| 67890   | @bot123  | "Breaking News!" | 1 (Bot)       | 0.87             |

---

## 🔥 **Future Enhancements**

🐬 Use **Deep Learning (LSTMs/BERT)** for better text analysis  
🐬 Add **Graph-based Features** (e.g., interaction networks)  
🐬 Implement **Real-time Bot Detection API**  

---

## 📜 **License**

This project is released under the **MIT License**. Feel free to modify and distribute it!

---

## 👨‍💻 **Contributing**

🚀 Contributions are welcome! Feel free to submit **Pull Requests (PRs)** or raise **Issues**.
