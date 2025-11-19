import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

FILE_PATH = "files/input/heart_disease.csv"
OUTPUT_FILE = "heart_disease.csv"

dataset = pd.read_csv(FILE_PATH)
y = dataset.pop("target")
x = dataset.copy()

x["thal"] = x["thal"].map(
    lambda val: "normal" if val not in ["fixed", "fixed", "reversible"] else val
)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

estimator = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

estimator.fit(x, y)

with open("estimator.pickle", "wb") as file:
    pickle.dump(estimator, file)

print("Model saved successfully as 'estimator.pickle'")

treated_df = pd.concat([x, y], axis=1)

treated_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccessfully created '{OUTPUT_FILE}' with the treated data.")