# Credit-Card-Fraud-Detection-using-Machine-Learning

This project is about detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, so special techniques like SMOTE and scaling were used to improve accuracy.

I trained and compared multiple ML models — Logistic Regression, Random Forest, Decision Tree, and SVM — to find which one performs best. The results were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

A simple web interface (using Flask) is also created where you can enter transaction details and check whether it is fraud or not. This project helped me understand real-world ML problems, especially with imbalanced data.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in September 2013.
- Total transactions: 284,807
- Fraudulent transactions: 492 (≈ 0.172%)

## Objective

- Build a machine learning model to classify transactions as **fraudulent** or **legitimate**.
- Handle **imbalanced data** effectively.
- Compare performance of different models.

## Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- imbalanced-learn (SMOTE)
- Flask (for frontend)
- Joblib (for model saving)

## Preprocessing Steps

- Features selected: `Time`, `V1` to `V28`, `Amount`
- Target: `Class` (0 = Non-fraud, 1 = Fraud)
- **StandardScaler** used to scale features
- **SMOTE** used to balance the dataset (oversampling minority class)
