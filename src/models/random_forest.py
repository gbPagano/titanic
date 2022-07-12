from src import src_path
import pandas as pd
from os import path


def random_forest():

    processed_path = path.join(src_path, "data", "processed")
    models_path = path.join(src_path, "models")

    validate = pd.read_csv(path.join(processed_path,"validate.csv"))

    final = pd.DataFrame(validate["PassengerId"])
    validate = validate.drop(columns=["PassengerId"])

    rf = train_rf()

    predicts = rf.predict(validate)
    final["Survived"] = pd.DataFrame(predicts)

    final.to_csv(path.join(models_path,"random_forest.csv"), index=False)

def train_rf():

    processed_path = path.join(src_path, "data", "processed")

    X = pd.read_csv(path.join(processed_path,"X_concatenated.csv"))
    y = pd.read_csv(path.join(processed_path,"y_concatenated.csv"))

    X = X.drop(columns=["PassengerId"])
    y = y.drop(columns=["PassengerId"])

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X, y)

    return rf

if __name__ == "__main__":
    random_forest()



