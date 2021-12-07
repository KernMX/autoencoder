from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Reshape
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.regularizers import L2
import statsmodels.api as sm
import numpy as np
from scipy.stats import ttest_ind

def get_threshold(errors, alpha=0.05):
    kde = sm.nonparametric.KDEUnivariate(errors)
    kde.fit()
    return kde.support[kde.cdf >= (1-alpha)][0]

def confidence(normal, anomalous, alpha=0.05):
    stat, pval = ttest_ind(normal, anomalous, axis=0)
    significance = pval < alpha
    return sum(significance)/len(significance)

class AutoencoderBase(tf.keras.Model, ABC):
    def __init__(self):
        super(AutoencoderBase, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

    @abstractmethod
    def build_architecture(self):
        raise NotImplementedError("Architecture build instructions unspecified.")

    def call(self, x):
        encoded = self.encoder(x)
        outputs = self.decoder(encoded)
        return outputs

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

class Autoencoder(AutoencoderBase):
    def __init__(self, input_dim, num_layers, lr, output_activation='sigmoid', activation='relu'):
        super(Autoencoder, self).__init__()
        self.build_architecture(input_dim, num_layers, lr, output_activation, activation)

    def build_architecture(self, input_dim, num_layers, lr, output_activation, activation):
        num_layers = min(num_layers, np.floor(np.log2(input_dim)))
        # Encoder
        for i in range(num_layers+1):
            self.encoder.add(Dense(input_dim//(2**i), activation=activation, kernel_regularizer=L2(lr)))
        # Decoder
        for i in range(num_layers-1,-1,-1):
            if i != 0:
                self.decoder.add(Dense(input_dim//(2**i), activation=activation, kernel_regularizer=L2(lr)))
            else:
                self.decoder.add(Dense(input_dim//(2**i), activation=output_activation, kernel_regularizer=L2(lr)))

class ConvolutionalAutoencoder(AutoencoderBase):
    def __init__(self, input_shape, output_activation='sigmoid', activation='relu'):
        super(ConvolutionalAutoencoder, self).__init__()
        self.build_architecture(input_shape, output_activation, activation)

    def build_architecture(self, input_shape, output_activation, activation):
        # Encoder
        self.encoder.add(Conv2D(32, 3, activation=activation, padding='same', input_shape=input_shape))
        self.encoder.add(MaxPooling2D(2))
        self.encoder.add(Conv2D(64, 3, activation=activation, padding='same'))
        self.encoder.add(MaxPooling2D(2))
        self.encoder.add(Conv2D(64, 3, activation=activation, padding='same'))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(49, activation=activation))
        self.encoder.add(Reshape((7,7,1)))
        # Decoder
        self.decoder.add(Conv2DTranspose(64, 3, activation=activation, padding='same', strides=2, input_shape=(7,7,1)))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2DTranspose(64, 3, activation=activation, padding='same', strides=2))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2DTranspose(32, 3, activation=activation, padding='same', strides=1))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2D(1, 3, activation=output_activation, padding='same', strides=1))

class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, input_shape, max_units, num_layers, lr, activation='relu'):
        super(LSTMAutoencoder, self).__init__()
        self.build_architecture(input_shape, max_units, num_layers, lr, activation)

    def build_architecture(self, input_shape, max_units, num_layers, lr, activation):
        # Encoder
        self.encoder.add(LSTM(16, input_shape=(input_shape[1], input_shape[2]), activation='relu', return_sequences=True, kernel_regularizer=L2(1e-1)))
        self.encoder.add(LSTM(4, activation='relu', return_sequences=False))
        self.encoder.add(RepeatVector(input_shape[1]))
        # Decoder
        self.decoder.add(LSTM(4, activation='relu', return_sequences=True))
        self.decoder.add(LSTM(16, activation='relu', return_sequences=True))
        self.decoder.add(TimeDistributed(Dense(input_shape[2])))
