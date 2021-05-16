import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from dataset import load_dataset


def calc_accuracy(model, data):
    param_names = model.metrics_names
    param_values = model.evaluate(data)
    result = {}
    for name, value in zip(param_names, param_values):
        result[name] = value
    return result


def evaluate(config_file="config.json"):
    # Loading the configuration file
    config = json.load(open(config_file, 'r'))

    # Loading the test_datagen
    _, _, test_generator = load_dataset()
    print(f"[INFO] Total Number of Test instances: {len(test_generator) * config['batch_size']}")

    # Loading the Saved Model
    model = load_model(os.path.join(config['checkpoint_filepath'], 'saved_model'))
    model.summary()

    # Calculating the Accuracy
    print(calc_accuracy(model, test_generator))


def plot_confusion_matrix(config_file="config.json"):
    # Loading the configuration file
    config = json.load(open(config_file, 'r'))

    # Loading the test_datagen
    _, _, test_generator = load_dataset()
    print(f"[INFO] Total Number of Test instances: {len(test_generator) * config['batch_size']}")

    # Loading the Saved Model
    model = load_model(os.path.join(config['checkpoint_filepath'], 'saved_model'))
    model.summary()

    # Generating Predictions on test images
    y_true, y_pred = [], []
    no_of_batches = len(test_generator)
    print(f"[INFO] Calculating predictions on test iamges")
    for i in range(no_of_batches):
        X, y = test_generator.next()
        y_pred_prob = model.predict(X)
        y_pred_batch = np.argmax(y_pred_prob, axis=1)
        y_true = y_true + list(y)
        y_pred = y_pred + list(y_pred_batch)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculating the confusion matrix
    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    print(con_mat_norm)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    # Plotting confusion matrix
    fig = plt.figure(figsize=(10,6))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()







if __name__ == "__main__":
    # evaluate()
    plot_confusion_matrix()
