import json
import os
import shutil
import textwrap

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report

from dataset import load_dataset

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


def find_misclassified(config_file="config.json"):
    # Loading the configuration file
    config = json.load(open(config_file, 'r'))

    # Loading the test_datagen
    _, _, test_generator = load_dataset()
    print(f"[INFO] Total Number of Test instances: {len(test_generator) * config['batch_size']}")

    # Loading the Saved Model
    model = load_model(os.path.join(config['checkpoint_filepath'], 'saved_model'))
    model.summary()

    # Removing the old directory or creating the new one
    classification_dir = os.path.join(config['checkpoint_filepath'], 'misclassified')
    if os.path.exists(classification_dir):
        shutil.rmtree(classification_dir)
        print(f"[INFO] Removing the old \'{classification_dir}\' directory")
        print(f"[INFO] Creating the new \'{classification_dir}\' directory")
        os.mkdir(classification_dir)
    else:
        print(f"[INFO] Creating the new \'{classification_dir}\' directory")
        os.mkdir(classification_dir)

    # Generating Predictions
    print(f"[INFO] Creating predictions...")
    pred_prob = model.predict(test_generator)
    y_pred = np.argmax(pred_prob, axis=1).astype(int)
    y_true = np.array(test_generator.classes).astype(int)
    class_labels = list(test_generator.class_indices.keys())
    file_paths = test_generator.filepaths
    print(f"[INFO] Prediction generation complete !")


    for prediction, ground_truth, img_url in zip(y_pred, y_true, file_paths):
        if prediction != ground_truth:
            new_filename = img_url.split(os.path.sep)[-1].replace('image ', '').replace('.JPG', '')
            img = cv2.imread(img_url)
            img = cv2.copyMakeBorder(img, 55, 0, 0, 0, cv2.BORDER_CONSTANT, None, [255, 255, 255])
            img = cv2.putText(img, f"Actual: {class_labels[ground_truth]}", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 0, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f"Prediction: {class_labels[prediction]}", (2, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(classification_dir, class_labels[ground_truth] + new_filename + '.jpg'), img)
            print(os.path.join(classification_dir, class_labels[ground_truth] + new_filename + '.jpg'))


if __name__ == "__main__":
    # evaluate()
    # plot_confusion_matrix()
    find_misclassified()
