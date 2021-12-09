import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from emnist import extract_training_samples, extract_test_samples

def load_emnist():
    trn_x, trn_y = extract_training_samples('digits')
    trn_y = np.ones_like(trn_y)
    trn_x = trn_x[:,:,:,np.newaxis] / 255
    normal_x, normal_y = extract_test_samples('digits')
    normal_y = np.ones_like(normal_y)
    anomaly_x, anomaly_y = extract_test_samples('letters')
    anomaly_y = np.zeros_like(anomaly_y)
    _, selected_x, _, selected_y = train_test_split(anomaly_x, anomaly_y, test_size=0.01)
    tst_x = np.concatenate([normal_x, selected_x], axis=0)
    tst_x = tst_x[:,:,:,np.newaxis] / 255
    tst_y = np.concatenate([normal_y, selected_y], axis=0)
    return trn_x, tst_x, trn_y, tst_y

def load_pump():
    # Clean the data
    df = pd.read_csv('sensor.csv')
    del df['sensor_15']
    del df['Unnamed: 0']
    df = df.drop_duplicates()
    df = df.dropna()
    df['date'] = pd.to_datetime(df['timestamp'])
    del df['timestamp']
    df = df.set_index('date')

    # Split data into train and test sets
    #df = df[df['machine_status'] != 'RECOVERING']
    trn_x = df[df['machine_status'] == 'NORMAL']
    trn_y = trn_x['machine_status']
    del trn_x['machine_status']

    tst_y = df['machine_status']
    del df['machine_status']
    tst_x = df

    trn_x = trn_x.values
    trn_y = trn_y.values
    tst_x = tst_x.values
    tst_y = tst_y.values

    # Change labels to numerics
    enc = OrdinalEncoder()
    tst_y = enc.fit_transform(tst_y.reshape(-1,1))
    trn_y = enc.transform(trn_y.reshape(-1,1))

    # Normalize the data
    scaler = MinMaxScaler()
    tst_x = scaler.fit_transform(tst_x)
    trn_x = scaler.transform(trn_x)

    # Dimensionality Reduction
    pca = PCA(0.95)
    tst_x = pca.fit_transform(tst_x)
    trn_x = pca.transform(trn_x)

    # Convert data to format for LSTM
    trn_x = to_timeseries(trn_x)
    tst_x = to_timeseries(tst_x)

    return trn_x, tst_x, trn_y, tst_y

def load_bearing():
    df = pd.read_csv('bearing_data.csv', index_col=0)
    trn_x = df['2004-02-12 10:52:39': '2004-02-15 12:52:39'].values
    tst_x = df['2004-02-15 12:52:39':].values

    # Normalize the data
    scaler = MinMaxScaler()
    trn_x = scaler.fit_transform(trn_x)
    tst_x = scaler.transform(tst_x)

    # reshape for timeseries LSTM
    trn_x = to_timeseries(trn_x)
    tst_x = to_timeseries(tst_x)

    return trn_x, tst_x, None, None

def to_timeseries(data):
    return data.reshape(data.shape[0], 1, data.shape[1])

def from_timeseries(data):
    return data.reshape(data.shape[0], data.shape[2])
