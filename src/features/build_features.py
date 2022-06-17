import pandas as pd
from sklearn import model_selection, impute, preprocessing
from sklearn.experimental import enable_iterative_imputer

def clean_titanic(df):
    df = df.drop(
        columns=[
            "name",
            "ticket",
            "home.dest",
            "boat",
            "body",
            "cabin",
        ]
    ).pipe(pd.get_dummies, drop_first=True)

    return df

def get_train_test_X_y(df, size=0.3):
    y = df.survived
    X = df.drop(columns="survived")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=size, random_state=42
    )
    num_cols = [
        "age",
        "fare",
    ]

    imputer = impute.IterativeImputer()
    X_train.loc[:,num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test.loc[:,num_cols] = imputer.transform(X_test[num_cols])

    cols = "pclass age sibsp parch fare sex_male embarked_Q embarked_S".split()
    sca = preprocessing.StandardScaler()

    X_train = sca.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = sca.fit_transform(X_test)
    X_test = pd.DataFrame(X_test, columns=cols)

    return X_train, X_test, pd.DataFrame(y_train), pd.DataFrame(y_test)