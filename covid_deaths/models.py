import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('./stored_models/transformer_model.pkl', 'rb'))
file_path = "./data/parsed_dataset.csv"
date_column = "Date_reported"
target_column = "New_deaths"
df = pd.read_csv(file_path, parse_dates=True)
df = df[[date_column, target_column, "New_cases_30_days_ago"]]
df = df.groupby(date_column).sum().reset_index()
df = df.dropna()
df[date_column] = pd.to_datetime(df.pop(date_column), format="%Y-%m-%d")
a = model.predict(df[[date_column]])
print(a)


