from transformers import (
    PretrainedConfig,
)
from gluonts.time_feature import (
    time_features_from_frequency_str,
)
from gluonts.itertools import Cyclic, Cached
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from typing import Iterable

import torch
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform.sampler import InstanceSampler
from typing import Optional

import numpy as np
import pandas as pd
from gluonts.time_feature import (
    time_features_from_frequency_str,
)
from gluonts.dataset.pandas import PandasDataset
from gluonts.itertools import Map
from datasets import Dataset, Features, Value, Sequence
from functools import lru_cache, partial
from gluonts.dataset.field_names import FieldName

from accelerate import Accelerator
def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    """
    Create a transformation pipeline for time series data based on the provided configuration.

    This function constructs a series of data transformations to preprocess time series data
    for a forecasting model. The transformations include removing unused fields, converting data
    to NumPy arrays, handling missing values, adding time-related features, and renaming fields
    to match expected input formats.

    Parameters:
    - freq (str): The frequency of the time series data (e.g., 'M' for monthly data).
    - config (PretrainedConfig): A configuration object that specifies the number and type of
      features in the time series data.

    Returns:
    - Transformation: A transformation pipeline that can be applied to time series data.
    """
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    """
    Creates an instance splitter for time series data in training, validation, or testing mode.
    It handles data windowing for model training or evaluation, considering the specified past and future lengths.

    Parameters:
    - config (PretrainedConfig): Configuration with prediction length, context length, etc.
    - mode (str): Operation mode - 'train', 'validation', or 'test'.
    - train_sampler (Optional[InstanceSampler]): Custom sampler for training (default: ExpectedNumInstanceSampler).
    - validation_sampler (Optional[InstanceSampler]): Custom sampler for validation (default: ValidationSplitSampler).

        Returns:
    - Transformation: Configured InstanceSplitter for the specified mode.
    """
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    """
    This function prepares a data loader for training a time series model. It involves transforming
    the data, creating training instances, and batching them for training.

    Parameters:
    - config (PretrainedConfig): Configuration specifying model details.
    - freq: Frequency of the time series data.
    - data: The dataset to be used for training.
    - batch_size (int): Size of each batch.
    - num_batches_per_epoch (int): Number of batches per epoch.
    - shuffle_buffer_length (Optional[int]): Buffer length for shuffling data (default: None).
    - cache_data (bool): Whether to cache data in memory (default: True).
    - **kwargs: Additional keyword arguments.

    Returns:
    - Iterable: An iterable over training batches.
    """
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream, is_train=True)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_test_dataloader(
    config: PretrainedConfig, freq, data, batch_size: int, **kwargs
) -> Iterable:
    """
    Creates a test data loader for time series forecasting.
    This function prepares a data loader for evaluating a time series model. It transforms the data,
    creates test instances, and batches them for model evaluation.

    Parameters:
    - config (PretrainedConfig): Configuration specifying model details.
    - freq: Frequency of the time series data.
    - data: The dataset to be used for testing.
    - batch_size (int): Size of each batch.
    - **kwargs: Additional keyword arguments.

    Returns:
    - Iterable: An iterable over test batches.

    """
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

class ProcessStartField:
    ts_id = 0

    def __call__(self, data):
        data["start"] = data["start"].to_timestamp()
        data["feat_static_cat"] = [self.ts_id]
        data["feat_dynamic_real"] = None
        # data["item_id"] = f"T{self.ts_id+1}"
        self.ts_id += 1

        return data
    
features = Features(
    {
        "start": Value("timestamp[s]"),
        "target": Sequence(Value("float32")),
        "feat_static_cat": Sequence(Value("uint64")),
        # "feat_static_real":  Sequence(Value("float32")),
        "feat_dynamic_real": Sequence(Sequence(Value("uint64"))),
        # "feat_dynamic_cat": Sequence(Sequence(Value("uint64"))),
        "item_id": Value("string"),
    }
)

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

class TransformerModel:

    def __init__(self, model, config, target_column, id_column, batch_size, frequency="D"):
        self.model = model
        self.config = config
        self.frequency = frequency
        self.target_column = target_column
        self.id_column = id_column
        self.batch_size = batch_size

    def predict(self, data):
        accelerator = Accelerator()
        device = accelerator.device
        unique_dates = data.index.unique()
        df_test = data[data.index >= unique_dates[-self.config.prediction_length]]
        ds_test = PandasDataset.from_long_dataframe(
            df_test, target=self.target_column, item_id=self.id_column
        )
        process_start = ProcessStartField()
        process_start.ts_id = 0
        list_ds_test = list(Map(process_start, ds_test))
        test_dataset = Dataset.from_list(list_ds_test, features=features)
        test_dataset.set_transform(partial(transform_start_field, freq=self.frequency))
        
        test_dataloader = create_test_dataloader(
            config=self.config,
            freq=self.frequency,
            data=test_dataset,
            batch_size=self.batch_size,
        )
        forecasts = []
        accelerator = Accelerator()
        device = accelerator.device

        for batch in test_dataloader:
            outputs = self.model.generate(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if self.config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if self.config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
            )
            forecasts.append(outputs.sequences.cpu().numpy())
        forecasts = np.vstack(forecasts)
        forecast_median = np.median(forecasts, 1)
        return forecast_median[0]