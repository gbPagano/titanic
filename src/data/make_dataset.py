import pandas as pd
from os import path

def make_processed():

    from src.features import clean_titanic, get_train_test_X_y
    from src import src_path

    raw_path = path.join(src_path, "data", "raw")
    processed_path = path.join(src_path, "data", "processed")

    df_raw = pd.read_csv(path.join(raw_path,"train.csv"))
    df_clean = clean_titanic(df_raw)
    X_train, X_test, y_train, y_test = get_train_test_X_y(df_clean)

    X_train.to_csv(path.join(processed_path,"X_train.csv"),index=False)
    X_test.to_csv(path.join(processed_path,"X_test.csv"),index=False)
    y_train.to_csv(path.join(processed_path,"y_train.csv"),index=False)
    y_test.to_csv(path.join(processed_path,"y_test.csv"),index=False)

if __name__ == "__main__":
    make_processed()
