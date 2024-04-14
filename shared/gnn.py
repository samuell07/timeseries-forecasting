import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from scipy.stats import skew, kurtosis

from shared.spektral_utilities import localpooling_filter

def get_timespan(df, today, days):    
    df = df[pd.date_range(today - timedelta(days=days), 
            periods=days, freq='D')] # day - n_days <= dates < day    
    return df

def create_features(df, today, seq_len, length):
    
    all_sequence = get_timespan(df, today, seq_len).values
    
    group_store = all_sequence.reshape((-1, length, seq_len))
    
    store_corr = np.stack([np.corrcoef(i) for i in group_store], axis=0)
    
    store_features = np.stack([
              group_store.mean(axis=2),
              group_store[:,:,int(seq_len/2):].mean(axis=2),
              group_store.std(axis=2),
              group_store[:,:,int(seq_len/2):].std(axis=2),
              skew(group_store, axis=2),
              kurtosis(group_store, axis=2),
              np.apply_along_axis(lambda x: np.polyfit(np.arange(0, seq_len), x, 1)[0], 2, group_store)
            ], axis=1)
    
    group_store = np.transpose(group_store, (0,2,1))
    store_features = np.transpose(store_features, (0,2,1))
    
    return group_store, store_corr, store_features

def create_label(df, today, length):
    
    y = df[today].values
    
    return y.reshape((-1, length))

class GNNModel:
    def __init__(self, models, date_column, target_column, id_column, sequence_length):
        self.models = models
        self.date_column = date_column
        self.target_column = target_column
        self.id_column = id_column
        self.sequence_length = sequence_length

    def predict(self, df):
        unique_dates = df[self.date_column].unique()
        um_countries_regions = len(df[self.id_column].unique())
        pivot_df = df.pivot_table(index=self.date_column, columns=self.id_column, values=self.target_column)

        df.rename(columns={"index": self.date_column}, inplace=True)
        unstaked_df = df.copy()
        unstaked_df["id"] = unstaked_df["WHO_region"]
        unstaked_df.set_index(["id", "Date_reported"], inplace=True)

        # Dropping columns not needed for the analysis
        unstaked_df.drop(["WHO_region"], axis=1, inplace=True)

        # Converting data to float and unstacking
        unstaked_df = unstaked_df.astype(float).unstack()
        unstaked_df.columns = unstaked_df.columns.get_level_values(1)
        test_date = unique_dates[-30]
        X_seq, X_cor, X_feat, y = [], [], [], []

        for d in tqdm(
            pd.date_range(test_date + timedelta(days=self.sequence_length), unique_dates[-1])
        ):
            seq_, corr_, feat_ = create_features(unstaked_df, d, self.sequence_length, um_countries_regions)
            y_ = create_label(unstaked_df, d, um_countries_regions)
            X_seq.append(seq_), X_cor.append(corr_), X_feat.append(feat_), y.append(y_)

        X_test_seq = np.concatenate(X_seq, axis=0).astype("float16")
        X_test_cor = np.concatenate(X_cor, axis=0).astype("float16")
        X_test_feat = np.concatenate(X_feat, axis=0).astype("float16")
        y_test = np.concatenate(y, axis=0).astype("float16")
        X_test_lap = localpooling_filter(1 - np.abs(X_test_cor))

        pred_test_all = np.zeros(y_test.shape)

        for region in range(um_countries_regions):
            
            model = self.models[region]

            pred_test_all[:, region] = model.predict(
                [X_test_seq, X_test_lap, X_test_feat]
            ).ravel()

        pred = np.sum(pred_test_all, axis=1)
        return pred