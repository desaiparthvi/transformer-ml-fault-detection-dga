import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_excel("data/dga_dataset.xlsx")
print(data.isnull().sum())
# Fill missing numeric values (gas columns)
data[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']] = data[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']].fillna(data[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']].mean())
# Remove rows where Fault Type is missing (cannot train without labels)
data = data.dropna(subset=['Fault Type'])
features = data[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']]
target = data['Fault Type']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Logistic Regression Results")
print(confusion_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Support Vector Machine
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_svm)
print("Support Vector Machine Results")
print(confusion_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_clf)
print("Random Forest Results")
print(confusion_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred_clf))
print("Classification Report:")
print(classification_report(y_test, y_pred_clf))

