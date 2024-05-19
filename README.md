# Leveraging Machine Learning for Time Series Predictive Analysis

This GitHub repository is a code base for a Diploma thesis.

## Installation

For running jupyter notebooks in the project (files which end with .ipynb), you will need to get the interface (https://docs.jupyter.org/en/latest/install.html).
Basics for running notebook: https://unidata.github.io/python-training/python/notebook/.

The requirements for this repository can be dowloaded with:
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

## Data Sources
The script fetches data from publicly available APIs:
- COVID-19 data from the World Health Organization.
- Bitcoin price data from Yahoo Finance.
- Electricity demand from https://www.eia.gov/

Ensure network access is available to fetch the latest data for predictions.

## Customizing the Script
The script is modular, allowing for easy expansion to include more datasets or models. To add a new dataset, define its structure in `dataset_metadata` and implement corresponding loading and preprocessing functions.

