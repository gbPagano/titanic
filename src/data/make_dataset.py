# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

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

    X_train.to_csv(path.join(processed_path,"X_train.csv"))
    X_test.to_csv(path.join(processed_path,"X_test.csv"))
    y_train.to_csv(path.join(processed_path,"y_train.csv"))
    y_test.to_csv(path.join(processed_path,"y_test.csv"))

if __name__ == "__main__":
    make_processed()
