import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv(r"C:\Users\Amaya\Downloads\creditcard.csv")
data.head(3)

# The data is already clean

features = np.array(data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                          'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                          'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
target = np.array(data['Class'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from imblearn.over_sampling import SMOTE
# After splitting into train/test
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)


from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(x_train_scaled, y_train)

# Predict
y_pred = model.predict(x_test_scaled)


from sklearn.metrics import confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities
y_probs = model.predict_proba(x_test_scaled)[:, 1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Plot
plt.plot(fpr, tpr, label="ROC Curve", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# AUC Score
print("AUC Score:", roc_auc_score(y_test, y_probs))

# import joblib
# joblib.dump(scaler, "scaler.pkl")  # Save the model

# joblib.dump(model, "fraud_model.pkl")  # Save the scaler (used to scale input data)
