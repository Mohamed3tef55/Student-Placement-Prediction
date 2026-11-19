import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

# Synthetic demo data: replace with the original student placement dataset.
rng = np.random.default_rng(22)
n = 700
df = pd.DataFrame({
    "cgpa": rng.uniform(5.0, 10.0, n).round(2),
    "internships": rng.integers(0, 4, n),
    "projects": rng.integers(0, 6, n),
    "communication": rng.integers(1, 6, n),
    "specialization": rng.choice(["IT", "Finance", "Marketing"], n),
})
score = (
    df["cgpa"] * 0.7 + df["internships"] * 0.6
    + df["projects"] * 0.25 + df["communication"] * 0.3
    + rng.normal(0, 1, n)
)
df["placed"] = (score > score.median()).astype(int)

X = df.drop(columns="placed")
y = df["placed"]
numeric_features = ["cgpa", "internships", "projects", "communication"]
categorical_features = ["specialization"]

preprocessor = ColumnTransformer([
    ("numeric", make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ), numeric_features),
    ("categorical", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ), categorical_features),
])

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))
df.to_csv(OUTPUTS / "demo_placement_data.csv", index=False)
print("Saved synthetic demo data to outputs/.")