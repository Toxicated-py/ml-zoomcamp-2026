import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def load_data():
    data_url = (
        "https://raw.githubusercontent.com/alexeygrigorev/"
        "mlbookcamp-code/master/chapter-03-churn-prediction/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(data_url)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    categorical_columns = list(
        df.select_dtypes(include=["object", "string"]).columns
    )
    for column in categorical_columns:
        df[column] = df[column].str.lower().str.replace(" ", "_")

    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
    df.totalcharges = df.totalcharges.fillna(0)
    df.churn = (df.churn == "yes").astype(int)

    return df


def train_model(df):
    numerical = ["tenure", "monthlycharges", "totalcharges"]
    categorical = [
        "gender",
        "seniorcitizen",
        "partner",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract",
        "paperlessbilling",
        "paymentmethod",
    ]

    y_train = df.churn
    train_dict = df[categorical + numerical].to_dict(orient="records")

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver="liblinear"),
    )
    pipeline.fit(train_dict, y_train)

    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, "wb") as f_out:
        pickle.dump(pipeline, f_out)


if __name__ == "__main__":
    df = load_data()
    pipeline = train_model(df)
    save_model(pipeline, "model.bin")
    print("Model saved to model.bin")
