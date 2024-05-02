# import pickle
# import numpy as np
# import pandas as pd

# model = pickle.load(open('./stored_models/transformer_model.pkl', 'rb'))
# file_path = "./data/parsed_dataset.csv"
# date_column = "Date_reported"
# target_column = "New_deaths"
# df = pd.read_csv(file_path, parse_dates=True)
# df = df[[date_column, target_column, "New_cases_30_days_ago"]]
# df = df.groupby(date_column).sum().reset_index()
# df = df.dropna()
# df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
# a = model.predict(df[[date_column]])
# print(a)

from datetime import datetime, timezone

import pandas as pd
import io
import requests
from scipy.ndimage import shift
import pytz
def load_covid_deaths():
    target_column = "New_deaths"
    date_column = "Date_reported"
    url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    df = df[df[target_column] >= 0]
    df = df[df['New_cases'] >= 0]
    df[target_column] = df[target_column] + 1
    # new_df = df.copy().iloc[30:]
    # new_column_cases = shift(df['New_cases'], shift=30)[30:]
    # new_df[f'New_cases_30_days_ago'] = new_column_cases
    # df = new_df.drop(columns=['New_cases', 'Cumulative_cases', 'Cumulative_deaths'])
    # df[date_column] = pd.to_datetime(df[date_column])
    # last_date = df[date_column].max()
    # date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    # new_df = pd.DataFrame(date_range, columns=[date_column])
    # new_df['New_cases_30_days_ago'] = df[target_column].iloc[-30:].values
    # df = pd.concat([df, new_df], ignore_index=True)
    return df

def load_btc():
    date_column = "Date"
    target_column = "Close"
    now = int(datetime.now(tz=pytz.utc).timestamp())
    url=f"https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1681404460&period2={now}&interval=1d&events=history"
    df = pd.read_csv(url)[[date_column, target_column]]
    new_df = df.copy().iloc[30:]
    new_column = shift(df[target_column], shift=30)[30:]
    new_df[f'Closed_30_days_ago'] = new_column
    df = new_df
    df[date_column] = pd.to_datetime(df[date_column])
    last_date = df[date_column].max()
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    new_df = pd.DataFrame(date_range, columns=[date_column])
    new_df['Closed_30_days_ago'] = df[target_column].iloc[-30:].values
    df = pd.concat([df, new_df], ignore_index=True)
    return df
# df = load_covid_deaths()
# firsst_date = df['Date_reported'].min()
# last_date = df['Date_reported'].max()
# date_range = pd.date_range(start=firsst_date, end=last_date, freq='D')
# new_df = pd.DataFrame(date_range, columns=['Date_reported'])

import numpy as np
def reindex_fill(group, date_range):
    group['New_deaths'].fillna(0, inplace=True)  # Set missing Value1 to 0
    group['Country_code'] = group['Country_code'].fillna(method='ffill')  # Forward fill the group name
    group['Country'] = group['Country'].fillna(method='ffill')  # Forward fill the group name
    group['WHO_region'] = group['WHO_region'].fillna(method='ffill')  # Forward fill the group name
    return group.reset_index()
target_column = "New_deaths"
date_column = "Date_reported"
url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
df[date_column] = pd.to_datetime(df[date_column])

all_dates = pd.date_range(start=df['Date_reported'].min(), end=df['Date_reported'].max(), freq='D')
groups = []
for name, group in df.groupby('Country_code'):
    df_all_dates = pd.DataFrame(all_dates, columns=['Date_reported'])
    country = group['Country'].iloc[0]
    who_region = group['WHO_region'].iloc[0]
    full_dates = pd.merge(df_all_dates, group, on='Date_reported', how='left')
    full_dates['Country_code'] = name
    full_dates['Country'] = country
    full_dates['WHO_region'] = who_region
    full_dates['New_deaths'].fillna(0, inplace=True)  # Set missing Value1 to 0
    groups.append(full_dates)
    break
new_df = pd.concat(groups, ignore_index=True)
print(new_df)
# filled_df = df.groupby('Country_code').apply(reindex_fill, date_range=all_dates).reset_index(drop=True)
new_df.to_csv('filled_covid_deaths.csv', index=False)
