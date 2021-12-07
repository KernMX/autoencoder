import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autoencoder import Autoencoder, LSTMAutoencoder, get_threshold, confidence
from preprocessing import load_emnist, load_pump, load_bearing, from_timeseries

def plot_loss(history):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()

def plot_decision(errors, threshold):
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Anomaly Reconstruction Loss Threshold', fontsize=16)
    sns.distplot(errors, bins=20, kde=True, color='blue', label='Reconstruction Loss');
    plt.axvline(x=threshold, color='red', label='Threshold')
    plt.legend(loc='upper right')
    plt.show()

def plot_threshold(errors, threshold):
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    plt.plot(errors, 'b', label='Reconstruction Loss', linewidth=2)
    plt.axhline(y=threshold, color='red', label='Threshold')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()

def plot_outliers(errors, threshold, true_y):
    anomalies = true_y[true_y == 2]
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    plt.plot(errors, 'b', label='Reconstruction Loss', linewidth=2)
    plt.axhline(y=threshold, color='red', label='Threshold')
    plt.yscale('log')
    plt.plot(anomalies, 'rx', linestyle='none', markersize=12)
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an Anomaly Detection Autoencoder.')
    parser.add_argument('model_name', type=str, nargs=1, help='Model save directory.')
    parser.add_argument('--emnist', nargs='?', const=True, default=False, help='Flag to utilize emnist data.')
    parser.add_argument('--bearing', nargs='?', const=True, default=False, help='Flag to utilize bearing data.')
    parser.add_argument('--pump', nargs='?', const=True, default=False, help='Flag to utilize pump sensor data.')
    parser.add_argument('--num_epochs', type=int, nargs='?', default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='Training batch size.')
    parser.add_argument('--val_split', type=float, nargs='?', default=0.05, help="Training loop validation split.")
    parser.add_argument('--notrain', nargs='?', const=True, default=False, help='Flag to skip the training step and utilize the pre-trained model at model_name.')
    args = parser.parse_args()

    # get command line arguments
    model_name = args.model_name[0]
    validation_split = args.val_split
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Load appropriate data and initialize autoencoder architecture
    if args.emnist:
        trn_x, tst_x, trn_y, tst_y = load_emnist()
        trn_x = trn_x.reshape(-1, trn_x.shape[1]*trn_x.shape[2])
        tst_x = tst_x.reshape(-1, tst_x.shape[1]*tst_x.shape[2])
        model = Autoencoder(input_dim=trn_x.shape[1], num_layers=3, lr=1e-1, output_activation='sigmoid', activation='relu')
    elif args.pump:
        trn_x, tst_x, trn_y, tst_y = load_pump()
        print(trn_x.shape)
        print(tst_x.shape)
        model = LSTMAutoencoder(input_shape=trn_x.shape, max_units=128, num_layers=3, lr=1e-1)
    elif args.bearing:
        trn_x, tst_x, trn_y, tst_y = load_bearing()
        model = LSTMAutoencoder(input_shape=trn_x.shape, max_units=16, num_layers=2, lr=1e-1)
    else:
        raise RuntimeError("You must specify a valid dataset.")

    # Train the autoencoder model
    if not args.notrain:
        # Train the model
        model.compile(optimizer='adam', loss='mae')
        history = model.fit(trn_x, trn_x, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split).history
        model.summary()
        plot_loss(history)
        model.save(f"saved_models/{model_name}")

    # Detect Outliers
    if args.notrain:
        # Load trained model
        model = tf.keras.models.load_model(f"saved_models/{model_name}")
        model.summary()

    # Calculate Anomaly threshold
    trn_reconstructions = model.predict(trn_x)
    if args.bearing or args.pump:
        trn_reconstructions = from_timeseries(trn_reconstructions)
        trn_x = from_timeseries(trn_x)
    errors = np.mean(np.abs(trn_reconstructions - trn_x), axis=1)
    threshold = get_threshold(errors, alpha=0.01)
    print(f"Threshold: {threshold}")
    plot_decision(errors, threshold)

    # Perform Anomaly Detection
    reconstructions = model.predict(tst_x)
    if args.bearing or args.pump:
        reconstructions = from_timeseries(reconstructions)
        tst_x = from_timeseries(tst_x)
    errors = np.mean(np.abs(reconstructions - tst_x), axis=1)
    plot_threshold(errors, threshold)
    msk = errors >= threshold
    anomalies = errors[msk]
    normals = tst_x[~msk]
    print(f"{len(anomalies)}/{len(anomalies)+len(normals)} entries are anomalous.")


    # Calculate the statistical confidence in anomalies
    metric = confidence(normals, anomalies, alpha=0.01)
    print(f"{metric*100}% confidence that anomalous data is statistically significant from normal data.")

    # Calculate the actual accuracy. precision, and recall
    if not args.bearing:
        plot_outliers(errors, threshold, tst_y)
        pred_y = [int(x) for x in msk]
        print(classification_report(tst_y, pred_y))
        matrix = confusion_matrix(tst_y, pred_y)
        plt.matshow(matrix)
        plt.show()
