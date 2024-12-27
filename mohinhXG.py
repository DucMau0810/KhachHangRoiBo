import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import category_encoders as ce
import xgboost as xgb
import joblib

customer_data = pd.read_csv("customer_churn.csv")

dataset = customer_data.copy()
     
he = ce.HashingEncoder(cols='state')
dataset_hash = he.fit_transform(dataset)
     
dataset_hash_dummy = pd.get_dummies(dataset_hash, drop_first=True)     

# View correlation
corr = dataset_hash_dummy.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=2)

# Remove correlation columns
new_dataset_hash_dummy_drop= dataset_hash_dummy.drop(columns=["voice_mail_plan_yes","total_day_charge","total_eve_charge","total_night_charge","total_intl_charge"])

#Huan luyen mo hinh
X = new_dataset_hash_dummy_drop.drop(["churn_yes"],axis=1)
y = new_dataset_hash_dummy_drop['churn_yes']

# Chia train ,test
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=42)

# XGBoost
model_xgb = xgb.XGBClassifier(random_state=42, n_estimators = 200)
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)

print(y_test)

print(y_pred)
# In bao cao ket qua
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion matrix: \n", conf_matrix)
print("Classification_report:\n", class_report)

# cm = confusion_matrix(y_test, y_pred)
# # Plot confusion matrix using seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix for XGBoost Model')
# plt.show()

#roc curve
# Tính giá trị FPR, TPR, và thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Tính diện tích dưới đường cong (AUC)
roc_auc = roc_auc_score(y_test, y_pred)

# Vẽ đường cong ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Đường chéo ngẫu nhiên
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

joblib.dump(model_xgb, 'xgboost_model.pkl')
print("Mo hinh da duoc luu")

