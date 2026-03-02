import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv("dataset_balanced.csv", header=None)

# clean
data = data.dropna()

# เอาเฉพาะแถวที่ column ครบ 43
data = data[data.apply(lambda x: len(x) == 43, axis=1)]


X = data.iloc[:, :-1]
y = data.iloc[:, -1] #label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("model.pkl ถูกสร้างแล้ว")
