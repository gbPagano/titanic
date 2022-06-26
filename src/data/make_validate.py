import pandas as pd
from os import path

def build_validate():

    from src import src_path
    from src.features import process_validate

    raw_path = path.join(src_path, "data", "raw")
    processed_path = path.join(src_path, "data", "processed")

    df_raw = pd.read_csv(path.join(raw_path,"train.csv"))

    df = process_validate(df_raw)

    df.to_csv(path.join(processed_path,"validate.csv"), index=False)


if __name__ == "__main__":
    build_validate()