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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
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


def evaluate(y_true=None, y_pred=None, config_dict=None):
    # Calculating the Accuracy
    acc_score = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    eval_report = '_____________________CLASSIFICATION REPORT____________________________\n'
    eval_report = eval_report + '\n' + f"Classification Accuracy: {acc_score}"
    eval_report = eval_report + '\n' + f"Mean Squared Error     : {mse}"
    eval_report = eval_report + '\n______________________________________________________________________\n'
    eval_report = eval_report + report
    print(type(report))
    print(eval_report)
    report_file_path = os.path.join(config_dict['checkpoint_filepath'], 'classification_report.txt')

    with open(report_file_path, "w") as outfile:
        outfile.write(eval_report)
        print(f"[INFO] Classification report is written in the file \'{report_file_path}\'")
        outfile.close()


def plot_confusion_matrix(y_true=None, y_pred=None, classes=None, config_dict=None):
    # Calculating the confusion matrix
    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    print("\n\n___________________ Confusion Matrix _____________________")
    print(con_mat_norm)
    print("______________________________________________________________")

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
    filepath = os.path.join(config_dict['checkpoint_filepath'], 'graphs',
                            f"4.confusion-matrix{config_dict['fig_format']}")
    plt.savefig(filepath)
    print(f"\n\n[INFO] Confusion Matrix is saved in \"{filepath}\"")

    # Calculating Classification Report.
    print("\n\n_________________Classification Report__________________")
    print(classification_report(y_true, y_pred))


def find_misclassified(y_true=None, y_pred=None, file_paths=None, config_dict=None):
    if config is None:
        print(f"\n\n[ERROR] No config dictionary found. \n Process Aborted!")
        return
    else:
        classification_dir = os.path.join(config_dict['checkpoint_filepath'], 'misclassified')

    # Removing the old directory or creating the new one
    if os.path.exists(classification_dir):
        shutil.rmtree(classification_dir)
        print(f"[INFO] Removing the old \'{classification_dir}\' directory")
        print(f"[INFO] Creating the new \'{classification_dir}\' directory")
        os.mkdir(classification_dir)
    else:
        print(f"[INFO] Creating the new \'{classification_dir}\' directory")
        os.mkdir(classification_dir)

    # Labeling the images and writing them
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
            print("Misclassified: ",
                  os.path.join(classification_dir, class_labels[ground_truth] + new_filename + '.jpg'))


if __name__ == "__main__":
    # Loading the configuration file
    config = json.load(open('config.json', 'r'))

    # Loading the test_datagen
    _, _, test_generator = load_dataset()
    print(f"[INFO] Total Number of Test instances: {len(test_generator) * config['batch_size']}")

    # Loading the Saved Model
    model = load_model(os.path.join(config['checkpoint_filepath'], 'saved_model'))
    model.summary()

    # Generating Predictions
    print(f"[INFO] Generating predictions...")
    pred_prob = model.predict(test_generator)
    y_pred = np.argmax(pred_prob, axis=1).astype(int)
    y_true = np.array(test_generator.classes).astype(int)

    # Getting the class names and file paths of the dataloader
    class_labels = list(test_generator.class_indices.keys())
    file_paths = test_generator.filepaths
    print(f"[INFO] Prediction generation complete !")

    evaluate(y_true=y_true, y_pred=y_pred, config_dict=config)
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=class_labels, config_dict=config)
    find_misclassified(y_true=y_true, y_pred=y_pred, file_paths=file_paths, config_dict=config)
