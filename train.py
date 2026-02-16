import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# โหลด dataset
data = pd.read_csv("dataset.csv", header=None)

X = data.iloc[:, :-1]   # landmark (42 ค่า)
y = data.iloc[:, -1]    # label (A,B,C,D)

# สร้าง model
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl ถูกสร้างแล้ว")
