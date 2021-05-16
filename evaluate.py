import os
import json
from keras.models import load_model
from dataset import load_dataset


def calc_accuracy(model, data):
    param_names = model.metrics_names
    param_values = model.evaluate(data)
    result = {}
    for name, value in zip(param_names, param_values):
        result[name] = value
    return result


def evaluate():
    # Loading the configuration file
    config = json.load(open("config.json", 'r'))

    # Loading the test_datagen
    _, _, test_generator = load_dataset()
    print(f"[INFO] Total Number of Test instances: {len(test_generator) * config['batch_size']}")

    # Loading the Saved Model
    model = load_model(os.path.join(config['checkpoint_filepath'], 'saved_model'))
    model.summary()

    # Calculating the Accuracy
    print(calc_accuracy(model, test_generator))


if __name__ == "__main__":
    evaluate()