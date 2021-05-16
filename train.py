import os
import json
import time
import numpy as np

from utils import print_config
from utils import load_callbacks
from utils import save_training_history
from utils import plot_training_summary
from dataset import load_dataset
from model import build_model


def run():
    # Loading the running configuration
    config = json.load(open("config.json", "r"))
    print_config(config)

    # Loading the dataloaders
    train_generator, valid_generator, test_generator = load_dataset()

    # Loading the model
    model = build_model()

    # Training the model
    start = time.time()
    train_history = model.fit(train_generator,
                              epochs=config["epochs"],
                              steps_per_epoch=len(train_generator),
                              validation_data=valid_generator,
                              validation_steps=len(valid_generator),
                              callbacks=load_callbacks(config))
    end = time.time()

    # Saving the model
    if not os.path.exists(config["checkpoint_filepath"]):
        print(f"[INFO] Creating directory {config['checkpoint_filepath']} to save the trained model")
        os.mkdir(config["checkpoint_filepath"])
    print(f"[INFO] Saving the model and log in \"{config['checkpoint_filepath']}\" directory")
    model.save(os.path.join(config["checkpoint_filepath"], 'saved_model'))

    # Saving the Training History
    save_training_history(train_history, config)

    # Plotting the Training History
    plot_training_summary(config)

    # Training Summary
    training_time_elapsed = end - start
    print(f"[INFO] Total Time elapsed: {training_time_elapsed} seconds")
    print(f"[INFO] Time per epoch: {training_time_elapsed//config['epochs']} seconds")


if __name__ == "__main__":
    run()