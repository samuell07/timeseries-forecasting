# timeseries-forecasting

## Installation

For running jupyter notebooks in the project (files which end with .ipynb), you will need to get the interface (https://docs.jupyter.org/en/latest/install.html).
Basics for running notebook: https://unidata.github.io/python-training/python/notebook/.

For all of the files above you need these libraries from python:

- pandas
- seaborn
- matplotlib
- sklearn
- numpy
- tensorflow

Or you can use this code:
```
pip install -r requirements.txt
```

# Time Series Prediction Script (tool.py)

## Overview
This Python script is capable of predicting data up to 30 days/hours (depending on data set).
The supported data sets are also included with whole research in their related folder


## Usage
To use this script, you will need to specify the model and dataset when running the script from the command line. Here is a basic example of how to invoke the script:

```bash
python timeseries_prediction.py --model lstm --dataset btc --forecastLength 15
```

### Parameters
- `--model (-m)`: The prediction model to use. Supported options include `regression`, `arima`, `sarima`, `lstm`, `prophet`, `prophet_log`, `transformer`, `dlt`, `ets`, `ktr`, `automl`, `gnn`.
- `--dataset (-d)`: The dataset to perform predictions on. Currently supports `covid_deaths` and `btc`.
- `--forecastLength (-fc)`: The length of the forecast, with a maximum value of 30 due to model limitations.

## Dependencies
Ensure the following Python libraries are installed before running the script:
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `scipy` for scientific computations like shifting series data.
- `pytz` for timezone calculations.
- `requests` for HTTP requests to fetch data.
- `argparse` for parsing command line arguments.
- `orbit`, `sklearn`, and other specific libraries depending on the chosen models.

## Data Sources
The script fetches data from publicly available APIs:
- COVID-19 data from the World Health Organization.
- Bitcoin price data from Yahoo Finance.

Ensure network access is available to fetch the latest data for predictions.

## Customizing the Script
The script is modular, allowing for easy expansion to include more datasets or models. To add a new dataset, define its structure in `dataset_metadata` and implement corresponding loading and preprocessing functions.

## Error Handling
The script includes error handling to manage common issues such as unsupported datasets or model configurations. For detailed error messages, ensure proper use of the command line arguments as per the usage instructions.
