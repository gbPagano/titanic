from src import src_path
import pandas as pd
from os import path


def random_forest():

    raw_path = path.join(src_path, "data", "raw")
    processed_path = path.join(src_path, "data", "processed")

    validate = pd.read_csv(path.join(processed_path,"validate.csv"))



    



