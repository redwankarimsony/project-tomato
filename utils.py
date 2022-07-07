import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Setting default fontsize and dpi
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300


def print_config(config_dict=None):
    for k, v in config_dict.items():
        if type(v) is dict:
            print('\t', k)
            for k_, v_ in v.items():
                print('\t\t', k_, ': ', v_)
        else:
            print('\t', k, ': ', v)


def load_callbacks(config):
    # Model Saving Checkpoints
    checkpoint_filepath = config["checkpoint_filepath"]
    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'model_snapshot.ckpt'),
                                                save_weights_only=False,
                                                monitor='val_acc',
                                                mode='max',
                                                save_best_only=True)

    # Early Stopper Callback
    early_stop = EarlyStopping(monitor='val_acc',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4)

    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, min_delta=1e-4)

    callbacks_list = [early_stop, reduce_lr, model_checkpoint_callback]
    return callbacks_list


def save_training_history(train_history, run_config):
    history = train_history.history
    df = pd.DataFrame(history)
    filepath = os.path.join(run_config["checkpoint_filepath"], 'train_log.csv')
    if os.path.exists(filepath):
        df.to_csv(filepath)
        print(
            f"[INFO] Training log is overwritten in {os.path.join(run_config['checkpoint_filepath'], 'train_log.csv')}")
    else:
        df.to_csv(filepath)
        print(f"[INFO] Training log is written in {os.path.join(run_config['checkpoint_filepath'], 'train_log.csv')}")


def plot_training_summary(run_config):
    if not os.path.exists(os.path.join(run_config['checkpoint_filepath'], 'train_log.csv')):
        print(f"[ERROR] Log file {os.path.join(run_config['checkpoint_filepath'], 'train_log.csv')} doesn't exist")
    else:
        df = pd.read_csv(os.path.join(run_config['checkpoint_filepath'], 'train_log.csv'))
        if not os.path.exists(os.path.join(run_config['checkpoint_filepath'], 'graphs')):
            os.mkdir(os.path.join(run_config['checkpoint_filepath'], 'graphs'))

        # Plotting the accuracy
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['acc'], "g*-", label="Training accuracy")
        plt.plot(df['val_acc'], "r*-", label="Validation accuracy")
        plt.title('Training and Validation Accuracy Graph')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid("both")
        plt.legend()
        plt.savefig(os.path.join(run_config['checkpoint_filepath'], 'graphs',
                                 f"1.accuracy-comparison{run_config['fig_format']}"))

        # Plotting the loss
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['loss'], "g*-", label="Training Loss")
        plt.plot(df['val_loss'], "r*-", label="Validation Loss")
        plt.title('Training and Validation Loss Graph')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid("both")
        plt.legend()
        plt.savefig(os.path.join(run_config['checkpoint_filepath'], 'graphs',
                                 f"2.loss-comparison{run_config['fig_format']}"))

        # Plotting the Learning Rate
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df['lr'], "b*-", label="Training Loss")
        plt.title('Training and Validation Loss Graph')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid("both")
        plt.legend()
        plt.savefig(os.path.join(run_config['checkpoint_filepath'],
                                 'graphs',
                                 f"3.learning-rate{run_config['fig_format']}"))


if __name__ == "__main__":
    # Testing the script
    default_config_file = 'config.json'
    config = json.load(open(default_config_file, 'r'))
    print_config(config)
    plot_training_summary(config)
