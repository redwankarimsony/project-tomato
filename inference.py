import os

from dataset import load_dataset
from utils import print_config
from model import build_model
import tensorflow as tf
from tensorflow import keras



train_generator, valid_generator, test_generator = load_dataset()
print("\n\n______________CLASS INDICES TO NAME MAPPING_______________")
print_config(train_generator.class_indices)
print("___________________________________________________________\n\n")


# model = build_model(config_file="inference-config.json")
# print(model.summary())


pretrained_model = keras.models.load_model(os.path.join("saved_models",
                                                                   "MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5" ))
print(pretrained_model.summary())

# predictions = tf.argmax(pretrained_model.predict(test_generator), axis=1)

result = pretrained_model.evaluate(test_generator)
print(result)

