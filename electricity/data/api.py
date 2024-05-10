import requests
import pandas as pd

def fetch_data(api_key, offset):
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


def save_df(df, chunk_size=2500000):
    df_size=len(df)
    for i, start in enumerate(range(0, df_size, chunk_size)):
        df[start:start+chunk_size].to_csv('electricity{}.csv'.format(i), index=False)

def main(api_key):
    offset = 0
    frames = []  # List to store data frames

    while True:
        print(f'Fetching data from offset: {offset}')
        data = fetch_data(api_key, offset)
        if data['response']['data']:
            df = pd.DataFrame(data['response']['data'])
            df = df[['subba', 'value', 'period', 'parent']]  # Select relevant columns
            frames.append(df)
            offset += 5000
        else:
            break

    result_df = pd.concat(frames, ignore_index=True)
    save_df(result_df)
    return result_df

api_key = '3zjKYxV86AqtJWSRoAECir1wQFscVu6lxXnRVKG8'
result_df = main(api_key)
print(result_df)
