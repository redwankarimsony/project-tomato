import os
import json
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from dataset import load_dataset
from sklearn.metrics import classification_report

# Setting default fontsize and dpi
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300


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
    print(f"[INFO] Calculating predictions on test images")
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
    print("\n\n___________________ Confusion Matrix _____________________")
    print(con_mat_norm)

    # Finding out the class names for labeling purpose
    classes = list(test_generator.class_indices.keys())
    classes = [x.replace('_', ' ') for x in classes]
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    # Plotting confusion matrix
    fig = plt.figure(figsize=(12, 10))
    ax = sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.set_xticklabels(textwrap.fill(x.get_text(), 20) for x in ax.get_xticklabels())
    ax.set_yticklabels(textwrap.fill(x.get_text(), 20) for x in ax.get_yticklabels())
    filepath = os.path.join(config['checkpoint_filepath'], 'graphs', '4.confusion-matrix.png')
    plt.savefig(filepath)
    print(f"\n\n[INFO] Confusion Matrix is saved in \"{filepath}\"")

    # Calculating Classification Report.
    print("\n\n_________________Classification Report__________________")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
    plot_confusion_matrix()
