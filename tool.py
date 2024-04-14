import datetime

import pytz
from shared.helpers import load_model
import argparse
import pandas as pd
import io
import requests
from scipy.ndimage import shift
import numpy as np
# for KTR model this import is needed
from orbit.models import KTR
from sklearn.preprocessing import StandardScaler
from shared.lstm import create_data
from shared.transformer import TransformerModel

def load_covid_deaths():
    target_column = "New_deaths"
    url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
    date_column = "Date_reported"
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode("utf-8")))
    df = df[df[target_column] >= 0]
    df = df[df["New_cases"] >= 0]
    df_max = df[target_column].max()
    df = df[df[target_column] != df_max]
    df[target_column] = df[target_column] + 1
    df[date_column] = pd.to_datetime(df[date_column])
    all_dates = pd.date_range(
        start=df["Date_reported"].min(),
        end=df["Date_reported"].max() + datetime.timedelta(30),
        freq="D",
    )
    groups = []
    for name, group in df.groupby("Country_code"):
        df_all_dates = pd.DataFrame(all_dates, columns=["Date_reported"])
        country = group["Country"].iloc[0]
        who_region = group["WHO_region"].iloc[0]
        full_dates = pd.merge(df_all_dates, group, on="Date_reported", how="left")
        full_dates["Country_code"] = name
        full_dates["Country"] = country
        full_dates["WHO_region"] = who_region
        full_dates["New_deaths"].fillna(1, inplace=True)
        full_dates["New_cases"].fillna(0, inplace=True)
        tmp_df = full_dates.copy().iloc[30:]
        new_column_cases = shift(full_dates["New_cases"], shift=30)[30:]
        tmp_df[f"New_cases_30_days_ago"] = new_column_cases
        groups.append(tmp_df)
    new_df = pd.concat(groups, ignore_index=True)
    df = new_df
    df = new_df.drop(columns=["New_cases", "Cumulative_cases", "Cumulative_deaths"])
    return df


def load_btc():
    date_column = "Date"
    target_column = "Close"
    now = int(datetime.now(tz=pytz.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1681404460&period2={now}&interval=1d&events=history"
    print(url)
    df = pd.read_csv(url, parse_dates=True)[[date_column, target_column]]
    new_df = df.copy().iloc[30:]
    new_column = shift(df[target_column], shift=30)[30:]
    new_df[f"Closed_30_days_ago"] = new_column
    df = new_df
    df.set_index(date_column, inplace=True, drop=False)
    return df


def load_dataset(dataset_name):
    if dataset_name == "covid_deaths":
        return load_covid_deaths()
    elif dataset_name == "btc":
        return load_btc()
        pass
    else:
        raise ValueError("Invalid dataset name")


def predict_arima(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, "New_cases_30_days_ago"]]
    df = df.groupby(date_column).sum().reset_index()
    df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
    return (
        model.predict(start=df[date_column].values[-30], end=df[date_column].values[-1])
        - 1
    )


def predict_regression(model, df):
    date_column = "Date_reported"
    df = df[[date_column, "New_cases_30_days_ago"]]
    df = df.groupby(date_column).sum().reset_index()
    df.set_index(date_column, inplace=True)
    df = df.select_dtypes(exclude=["object"])
    return model.predict(df.iloc[-30:]) - 1


def predict_bayes(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, "New_cases_30_days_ago"]]
    df = df.groupby(date_column).sum().reset_index()
    df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
    return model.predict(df.iloc[-30:])["prediction"] - 1

def predict_prohpeth(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, "New_cases_30_days_ago"]]
    df = df.groupby(date_column).sum().reset_index()
    df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
    df = df.rename(columns={date_column: "ds", target_column: "y"})
    return model.predict(df.iloc[-30:][['ds']])["yhat"] - 1

def predict_prohpeth_log(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, "New_cases_30_days_ago"]]
    df = df.groupby(date_column).sum().reset_index()
    df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
    df = df.rename(columns={date_column: "ds", target_column: "y"})
    return np.exp(model.predict(df.iloc[-30:][['ds']])["yhat"]) - 1

def predict_automl(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, 'New_cases_30_days_ago']]
    df[date_column] = pd.to_datetime(df[date_column], format="%Y-%m-%d")


    df = df.groupby(date_column).sum().reset_index()
    df.set_index(date_column, inplace=True)
    print(df)
    model.fit_data(df.iloc[:-30])
    return (model.predict(forecast_length=1).forecast[target_column] - 1)[:30]

def predict_lstm(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"
    df = df[[date_column, target_column, "New_cases_30_days_ago"]]

    df = df.groupby(date_column).sum().reset_index()
    df.set_index(date_column, inplace=True)
    df = df.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    _, X_test, _, _, _, _ = create_data(
        scaled_data,
        n_future=1,
        n_past=30,
        train_test_split_percentage=0.9,
        validation_split_percentage=0,
        prediction_length=30,
    )
    y_pred = model.predict(X_test)
    scaler = StandardScaler()
    scaler.fit(df[[target_column]])
    unscaled_y_pred = scaler.inverse_transform([y_pred.flatten()])[0]
    return unscaled_y_pred - 1

def predict_transformer(model, df):
    date_column = "Date_reported"
    target_column = "New_deaths"

    df = df.groupby(date_column).sum().reset_index()
    df = df.dropna()

    df["id"] = "id"
    df[date_column] = pd.to_datetime(df[date_column])
    df[target_column] = df[target_column].fillna(0)
    df.set_index(date_column, inplace=True)
    return model.predict(df)

def predict(model, df, forecast_length, model_name):
    if model_name == "arima":
        return predict_arima(model, df)
    elif model_name == "sarima":
        return predict_arima(model, df)
    elif model_name == "regression":
        return predict_regression(model, df)
    elif model_name == "ets" or model_name == "ktr" or model_name == "dlt":
        return predict_bayes(model, df)
    elif model_name == "prophet":
        return predict_prohpeth(model, df)
    elif model_name == "prophet_log":
        return predict_prohpeth_log(model, df)
    elif model_name == "automl":
        return predict_automl(model, df)
    elif model_name == "lstm":
        return predict_lstm(model, df)
    elif model_name == "transformer":
        return predict_transformer(model, df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="The model to use for prediction, best used otherwise",
        choices=[
            "regression",
            "arima",
            "sarima",
            "lstm",
            "prophet",
            "prophet_log",
            "transformer",
            "dlt",
            "ets",
            "ktr",
            "automl"
        ],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="The dataset to use for prediction",
        choices=["covid_deaths", "btc"],
        required=True,
    )
    parser.add_argument(
        "-fc",
        "--forecastLength",
        type=int,
        help="The forecasting length, up to 30 due to model limitations",
        default=30,
    )
    args = parser.parse_args()
    model_name = args.model
    if args.forecastLength > 30:
        raise ValueError("The forecast length cannot exceed 30")
    if not model_name:
        if args.dataset == "covid_deaths":
            model_name = "prophet"
        elif args.dataset == "btc":
            model_name = "lstm"
    dataset = load_dataset(args.dataset)
    model = load_model(model_name, args.dataset)
    predictions = predict(model, dataset, args.forecastLength, model_name)
    for i in predictions:
        print(i)


if __name__ == "__main__":
    main()
