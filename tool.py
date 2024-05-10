from datetime import datetime, timedelta

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
from shared.gnn import GNNModel

dataset_metadata = {
    "covid_deaths": {
        "date_column": "Date_reported",
        "target_column": "New_deaths",
        "id_column": "WHO_region",
        "sequence_length": 14,
        "extra_columns": ["New_cases_30_days_ago"],
        "freq": "D"
    },
    "btc": {
        "date_column": "Date",
        "target_column": "Close",
        "id_column": None,
        "sequence_length": 14,
        "extra_columns": ["Closed_30_days_ago"],
        "freq": "D"
    },
    "electricity": {
        "date_column": "period",
        "target_column": "value",
        "id_column": "parent",
        "sequence_length": 24,
        "extra_columns": ["value_30_hours_ago"],
        "freq": "H"
    },
}


def load_covid_deaths():
    url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode("utf-8")))
    df = df[df[dataset_metadata["covid_deaths"]["target_column"]] >= 0]
    df = df[df["New_cases"] >= 0]
    df_max = df[dataset_metadata["covid_deaths"]["target_column"]].max()
    df = df[df[dataset_metadata["covid_deaths"]["target_column"]] != df_max]
    df[dataset_metadata["covid_deaths"]["target_column"]] = (
        df[dataset_metadata["covid_deaths"]["target_column"]] + 1
    )
    df[dataset_metadata["covid_deaths"]["date_column"]] = pd.to_datetime(
        df[dataset_metadata["covid_deaths"]["date_column"]]
    )
    all_dates = pd.date_range(
        start=df["Date_reported"].min(),
        end=df["Date_reported"].max() + timedelta(30),
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
    now = int(datetime.now(tz=pytz.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1681404460&period2={now}&interval=1d&events=history"
    print(url)
    df = pd.read_csv(url, parse_dates=True)[
        [
            dataset_metadata["btc"]["date_column"],
            dataset_metadata["btc"]["target_column"],
        ]
    ]
    new_df = df.copy().iloc[30:]
    new_column = shift(df[dataset_metadata["btc"]["target_column"]], shift=30)[30:]
    new_df[f"Closed_30_days_ago"] = new_column
    df = new_df
    print(df)
    return df

def fetch_ele_data(api_key, offset):
    url = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
    params = {
        'api_key': api_key,
        'frequency': 'hourly',
        'data[0]': 'value',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': offset,
        'length': 5000
    }
    response = requests.get(url, params=params)
    return response.json()

def load_eletricity():
    api_key = '3zjKYxV86AqtJWSRoAECir1wQFscVu6lxXnRVKG8'
    date_column = "period"
    target_column = "value"
    offset = 0
    frames = []  # List to store data frames
    date_str = None
    print(f'Fetching data from offset: {offset}')
    data = fetch_ele_data(api_key, offset)
    df = pd.DataFrame(data['response']['data'])
    df = df[['subba', 'value', 'period', 'parent']]  # Select relevant columns
    df['value'] = df['value'].astype(int)


    df = df.groupby([date_column, 'parent']).sum().reset_index()
    df.set_index([date_column, 'parent'], inplace=True)
    df.drop(columns=['subba'], inplace=True)
    df.reset_index(inplace=True)
    df[date_column] = pd.to_datetime(df[date_column])
    df_by_parent = df.groupby('parent')
    all_dates = pd.date_range(start=df[date_column].min(), end=df[date_column].max()+timedelta(hours=30), freq='H')
    groups = []
    for name, group in df_by_parent:
        df_all_dates = pd.DataFrame(all_dates, columns=[date_column])
        full_dates = pd.merge(df_all_dates, group, on=date_column, how='left')
        full_dates['parent'] = name
        full_dates[target_column].fillna(0, inplace=True)
        tmp_df = full_dates.copy().iloc[30:]
        new_column_cases = shift(full_dates[target_column], shift=30)[30:]
        tmp_df['value_30_hours_ago'] = new_column_cases
        groups.append(tmp_df)
    new_df = pd.concat(groups, ignore_index=True)
    new_df[target_column] = new_df[target_column].astype(int)
    new_df[target_column] += 1
    new_df['value_30_hours_ago'] = new_df['value_30_hours_ago'].astype(int)
    new_df['value_30_hours_ago'] += 1
    print(new_df)
    return new_df


def load_dataset(dataset_name):
    if dataset_name == "covid_deaths":
        return load_covid_deaths()
    elif dataset_name == "btc":
        return load_btc()
    elif dataset_name == "electricity":
        return load_eletricity()
    else:
        raise ValueError("Invalid dataset name")


def predict_arima(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df.pop(dataset_metadata[dataset_name]["date_column"]), format="%Y-%m-%d"
    )
    return (
        model.predict(
            start=df[dataset_metadata[dataset_name]["date_column"]].values[-30],
            end=df[dataset_metadata[dataset_name]["date_column"]].values[-1],
        )
        - decrease
    )


def predict_regression(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df.set_index(dataset_metadata[dataset_name]["date_column"], inplace=True)
    df = df.select_dtypes(exclude=["object"])
    return model.predict(df.iloc[-30:]) - decrease


def predict_bayes(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df.pop(dataset_metadata[dataset_name]["date_column"]), format="%Y-%m-%d"
    )
    return model.predict(df.iloc[-30:])["prediction"] - decrease


def predict_prohpeth(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df.pop(dataset_metadata[dataset_name]["date_column"]), format="%Y-%m-%d"
    )
    df = df.rename(
        columns={
            dataset_metadata[dataset_name]["date_column"]: "ds",
            dataset_metadata[dataset_name]["target_column"]: "y",
        }
    )
    return model.predict(df.iloc[-30:][["ds"]])["yhat"] - decrease


def predict_prohpeth_log(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df.pop(dataset_metadata[dataset_name]["date_column"]), format="%Y-%m-%d"
    )
    df = df.rename(
        columns={
            dataset_metadata[dataset_name]["date_column"]: "ds",
            dataset_metadata[dataset_name]["target_column"]: "y",
        }
    )
    return np.exp(model.predict(df.iloc[-30:][["ds"]])["yhat"]) - decrease


def predict_automl(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df[dataset_metadata[dataset_name]["date_column"]], format="%Y-%m-%d"
    )

    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df.set_index(dataset_metadata[dataset_name]["date_column"], inplace=True)
    print(df)
    model.fit_data(df.iloc[:-30])
    return (
        model.predict(forecast_length=1).forecast[
            dataset_metadata[dataset_name]["target_column"]
        ]
        - decrease
    )[:30]


def predict_lstm(model, df, dataset_name, decrease):
    df = df[
        [
            dataset_metadata[dataset_name]["date_column"],
            dataset_metadata[dataset_name]["target_column"],
            *dataset_metadata[dataset_name]["extra_columns"],
        ]
    ]

    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df.set_index(dataset_metadata[dataset_name]["date_column"], inplace=True)
    df = df.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    _, X_test, _, _, _, _ = create_data(
        scaled_data,
        n_future=1,
        n_past=24,
        train_test_split_percentage=0.9,
        validation_split_percentage=0,
        prediction_length=30,
    )
    y_pred = model.predict(X_test)
    scaler = StandardScaler()
    scaler.fit(df[[dataset_metadata[dataset_name]["target_column"]]])
    unscaled_y_pred = scaler.inverse_transform([y_pred.flatten()])[0][-30:]
    return unscaled_y_pred - decrease


def predict_transformer(model, df, dataset_name, decrease):
    df = df.groupby(dataset_metadata[dataset_name]["date_column"]).sum().reset_index()
    df = df.dropna()

    df["item_id"] = "id"
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df[dataset_metadata[dataset_name]["date_column"]]
    )
    df[dataset_metadata[dataset_name]["target_column"]] = df[
        dataset_metadata[dataset_name]["target_column"]
    ].fillna(0)
    df.set_index(dataset_metadata[dataset_name]["date_column"], inplace=True)
    return (model.predict(df) - decrease)[-30:]


def predict_gnn(model, df, dataset_name, decrease):
    df[dataset_metadata[dataset_name]["date_column"]] = pd.to_datetime(
        df[dataset_metadata[dataset_name]["date_column"]]
    )
    df = (
        df.groupby(
            [
                dataset_metadata[dataset_name]["date_column"],
                dataset_metadata[dataset_name]["id_column"],
            ]
        )[dataset_metadata[dataset_name]["target_column"]]
        .sum()
        .reset_index()
    )
    return (model.predict(df))

def load_gnn_model(dataset_name):
    return GNNModel(
        date_column=dataset_metadata[dataset_name]["date_column"],
        target_column=dataset_metadata[dataset_name]["target_column"],
        id_column=dataset_metadata[dataset_name]["id_column"],
        sequence_length=dataset_metadata[dataset_name]["sequence_length"],
        freq=dataset_metadata[dataset_name]["freq"],
        dataset=dataset_name,
    )


def predict(model, df, model_name, dataset_name, decrease):
    if model_name == "arima" or model_name == "sarima":
        return predict_arima(model, df, dataset_name, decrease)
    elif model_name == "regression":
        return predict_regression(model, df, dataset_name, decrease)
    elif model_name == "ets" or model_name == "ktr" or model_name == "dlt":
        return predict_bayes(model, df, dataset_name, decrease)
    elif model_name == "prophet":
        return predict_prohpeth(model, df, dataset_name, decrease)
    elif model_name == "prophet_log":
        return predict_prohpeth_log(model, df, dataset_name, decrease)
    elif model_name == "automl":
        return predict_automl(model, df, dataset_name, decrease)
    elif model_name == "lstm":
        return predict_lstm(model, df, dataset_name, decrease)
    elif model_name == "transformer":
        return predict_transformer(model, df, dataset_name, decrease)
    elif model_name == "gnn":
        return predict_gnn(model, df, dataset_name, decrease)


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
            "automl",
            "gnn",
        ],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="The dataset to use for prediction",
        choices=["covid_deaths", "btc", "electricity"],
        # required=True,
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
        elif args.dataset == "electricity":
            model_name = "lstm"
    dataset = load_dataset(args.dataset)
    decrease = 0 if args.dataset == "btc" else 1
    if model_name == "gnn":
        if args.dataset == "btc":
            raise ValueError("GNN model is not supported for this dataset")
        model = load_gnn_model(args.dataset)
    else:
        model = load_model(model_name, args.dataset)
    predictions = predict(model, dataset, model_name, args.dataset, decrease)
    for index, value in enumerate(predictions):
        if index >= args.forecastLength:
            break
        print(value)


if __name__ == "__main__":
    main()
