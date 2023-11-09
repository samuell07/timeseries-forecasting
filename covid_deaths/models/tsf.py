from datetime import datetime
from distutils.util import strtobool

import numpy as np
import pandas as pd


# Converts the contents in a .tsf file into a dataframe and returns
# it along with other meta-data of the dataset:
# frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:  # Attributes have both name and type
                                raise ValueError("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise ValueError("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise ValueError("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise ValueError("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise ValueError("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise ValueError("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise ValueError(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise ValueError(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(np.array(numeric_series, dtype=np.float32))

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                            else:
                                raise ValueError(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise ValueError("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise ValueError("Empty file.")
        if len(col_names) == 0:
            raise ValueError("Missing attribute section.")
        if not found_data_section:
            raise ValueError("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def convert_multiple(text: str) -> str:
    if text.isnumeric():
        return text
    if text == "half":
        return "0.5"


def frequency_converter(freq: str):
    parts = freq.split("_")
    if len(parts) == 1:
        return BASE_FREQ_TO_PANDAS_OFFSET[parts[0]]
    if len(parts) == 2:
        return convert_multiple(parts[0]) + BASE_FREQ_TO_PANDAS_OFFSET[parts[1]]
    raise ValueError(f"Invalid frequency string {freq}.")


BASE_FREQ_TO_PANDAS_OFFSET = {
    "seconds": "S",
    "minutely": "T",
    "minutes": "T",
    "hourly": "H",
    "hours": "H",
    "daily": "D",
    "days": "D",
    "weekly": "W",
    "weeks": "W",
    "monthly": "M",
    "months": "M",
    "quarterly": "Q",
    "quarters": "Q",
    "yearly": "Y",
    "years": "Y",
}

# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")

# print(loaded_data)
# print(frequency)
# print(forecast_horizon)
# print(contain_missing_values)
# print(contain_equal_length)


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from pandas.tseries.frequencies import to_offset

import datasets
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{godahewa2021monash,
    author = "Godahewa, Rakshitha and Bergmeir, Christoph and Webb, Geoffrey I. and Hyndman, Rob J. and Montero-Manso, Pablo",
    title = "Monash Time Series Forecasting Archive",
    booktitle = "Neural Information Processing Systems Track on Datasets and Benchmarks",
    year = "2021",
    note = "forthcoming"
}
"""

_DESCRIPTION = """\
Monash Time Series Forecasting Repository which contains 30+ datasets of related time series for global forecasting research. This repository includes both real-world and competition time series datasets covering varied domains.
"""

_HOMEPAGE = "https://forecastingdata.org/"

_LICENSE = "The Creative Commons Attribution 4.0 International License. https://creativecommons.org/licenses/by/4.0/"


@dataclass
class MonashTSFBuilderConfig(datasets.BuilderConfig):
    """MonashTSF builder config with some added meta data."""

    file_path: Optional[str] = None
    prediction_length: Optional[int] = None
    item_id_column: Optional[str] = None
    data_column: Optional[str] = None
    target_fields: Optional[List[str]] = None
    feat_dynamic_real_fields: Optional[List[str]] = None
    multivariate: bool = False
    rolling_evaluations: int = 1


class LocalTSF(datasets.GeneratorBasedBuilder):
    """Builder of Monash Time Series Forecasting repository of datasets."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = MonashTSFBuilderConfig

    BUILDER_CONFIGS = [
        MonashTSFBuilderConfig(
            name="covid_deaths_local",
            version=VERSION,
            description="Local TSF file for COVID-19 deaths",
            file_path="../data/covid_deaths_dataset.tsf",
            feat_dynamic_real_fields=["total_cases", "new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "new_vaccinations", "stringency_index", "population", "gdp_per_capita", "hospital_beds_per_thousand"]
            # You need to specify the other config parameters as per your dataset's needs
        ),
    ]

    def _info(self):
        if self.config.multivariate:
            features = datasets.Features(
                {
                    "start": datasets.Value("timestamp[s]"),
                    "target": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    "feat_static_cat": datasets.Sequence(datasets.Value("uint64")),
                    # "feat_static_real":  datasets.Sequence(datasets.Value("float32")),
                    "feat_dynamic_real": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    # "feat_dynamic_cat": datasets.Sequence(datasets.Sequence(datasets.Value("uint64"))),
                    "item_id": datasets.Value("string"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "start": datasets.Value("timestamp[s]"),
                    "target": datasets.Sequence(datasets.Value("float32")),
                    "feat_static_cat": datasets.Sequence(datasets.Value("uint64")),
                    # "feat_static_real":  datasets.Sequence(datasets.Value("float32")),
                    "feat_dynamic_real": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    # "feat_dynamic_cat": datasets.Sequence(datasets.Sequence(datasets.Value("uint64"))),
                    "item_id": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        file_path = self.config.file_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": file_path, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path,
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        (
            loaded_data,
            frequency,
            forecast_horizon,
            _,
            _,
        ) = convert_tsf_to_dataframe(filepath, value_column_name="target")

        if forecast_horizon is None:
            prediction_length_map = {
                "S": 60,
                "T": 60,
                "H": 48,
                "D": 30,
                "W": 8,
                "M": 12,
                "Y": 4,
            }
            freq = frequency_converter(frequency)
            freq = to_offset(freq).name
            forecast_horizon = prediction_length_map[freq]

        if self.config.prediction_length is not None:
            forecast_horizon = self.config.prediction_length

        if self.config.item_id_column is not None:
            loaded_data.set_index(self.config.item_id_column, inplace=True)
            loaded_data.sort_index(inplace=True)

            for cat, item_id in enumerate(loaded_data.index.unique()):
                ts = loaded_data.loc[item_id]
                start = ts.start_timestamp[0]

                if self.config.target_fields is not None:
                    target_fields = ts[ts[self.config.data_column].isin(self.config.target_fields)]
                else:
                    target_fields = self.config.data_column.unique()

                if self.config.feat_dynamic_real_fields is not None:
                    feat_dynamic_real_fields = ts[
                        ts[self.config.data_column].isin(self.config.feat_dynamic_real_fields)
                    ]
                    feat_dynamic_real = np.vstack(feat_dynamic_real_fields.target)
                else:
                    feat_dynamic_real = None

                target = np.vstack(target_fields.target)

                feat_static_cat = [cat]

                if split in ["train", "val"]:
                    offset = forecast_horizon * self.config.rolling_evaluations + forecast_horizon * (split == "train")
                    target = target[..., :-offset]
                    if self.config.feat_dynamic_real_fields is not None:
                        feat_dynamic_real = feat_dynamic_real[..., :-offset]

                yield cat, {
                    "start": start,
                    "target": target,
                    "feat_dynamic_real": feat_dynamic_real,
                    "feat_static_cat": feat_static_cat,
                    "item_id": item_id,
                }
        else:
            if self.config.target_fields is not None:
                target_fields = loaded_data[loaded_data[self.config.data_column].isin(self.config.target_fields)]
            else:
                target_fields = loaded_data
            if self.config.feat_dynamic_real_fields is not None:
                feat_dynamic_real_fields = loaded_data[
                    loaded_data[self.config.data_column].isin(self.config.feat_dynamic_real_fields)
                ]
            else:
                feat_dynamic_real_fields = None

            for cat, ts in target_fields.iterrows():
                start = ts.get("start_timestamp", datetime.strptime("1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"))
                target = ts.target
                if feat_dynamic_real_fields is not None:
                    feat_dynamic_real = np.vstack(feat_dynamic_real_fields.target)
                else:
                    feat_dynamic_real = None

                feat_static_cat = [cat]
                if self.config.data_column is not None:
                    item_id = f"{ts.series_name}-{ts[self.config.data_column]}"
                else:
                    item_id = ts.series_name

                if split in ["train", "val"]:
                    offset = forecast_horizon * self.config.rolling_evaluations + forecast_horizon * (split == "train")
                    target = target[..., :-offset]
                    if feat_dynamic_real is not None:
                        feat_dynamic_real = feat_dynamic_real[..., :-offset]

                yield cat, {
                    "start": start,
                    "target": target,
                    "feat_dynamic_real": feat_dynamic_real,
                    "feat_static_cat": feat_static_cat,
                    "item_id": item_id,
                }
