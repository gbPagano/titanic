import pandas as pd
from sklearn import model_selection, impute, preprocessing
from sklearn.experimental import enable_iterative_imputer

def clean_titanic(df):
    df = df.drop(
        columns=[
            "Name",
            "Ticket",
            "Cabin",
        ]
    ).pipe(pd.get_dummies, drop_first=True)

    return df

def get_train_test_X_y(df, size=0.3):
    y = df[["PassengerId","Survived"]]
    X = df.drop(columns="Survived")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=size, random_state=42
    )
    num_cols = [
        "Age",
        "Fare",
    ]

    imputer = impute.IterativeImputer()
    X_train.loc[:,num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test.loc[:,num_cols] = imputer.transform(X_test[num_cols])


    cols = "Pclass,Age,SibSp,Parch,Fare".split(",")
    sca = preprocessing.StandardScaler()

    X_train.loc[:,cols] = sca.fit_transform(X_train[cols])
    #X_train = pd.DataFrame(X_train, columns=cols)
    X_test.loc[:,cols] = sca.fit_transform(X_test[cols])
    #X_test = pd.DataFrame(X_test, columns=cols)

    return X_train, X_test, pd.DataFrame(y_train), pd.DataFrame(y_test)