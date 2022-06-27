import pandas as pd
from os import path

def concatenated():

    from src import src_path

    processed_path = path.join(src_path, "data", "processed")

    X_train = pd.read_csv(path.join(processed_path,"X_train.csv"))
    X_test = pd.read_csv(path.join(processed_path,"X_test.csv"))
    y_train = pd.read_csv(path.join(processed_path,"y_train.csv"))
    y_test = pd.read_csv(path.join(processed_path,"y_test.csv"))

    X = pd.concat([X_train,X_test])
    y = pd.concat([y_train,y_test])

    X.to_csv(path.join(processed_path,"X_concatenated.csv"),index=False)
    y.to_csv(path.join(processed_path,"y_concatenated.csv"),index=False)


if __name__ == "__main__":
    concatenated()


