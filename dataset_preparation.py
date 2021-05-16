import json
import os
import glob
import pandas as pd


def get_dataframe(filelist=None, labels_dict=None):
    labels = sorted(list(labels_dict.keys()))
    if filelist is None:
        print(f"[ERROR]: No files found in the list")
        return None
    else:
        filenames = []
        labels_idx = []
        labels_name = []
        for filepath in filelist:
            filenames.append(filepath)
            labels_name.append(filepath.split(os.path.sep)[3].split('___')[1])
            labels_idx.append(str(labels.index(filepath.split(os.path.sep)[3].split('___')[1])))
        return pd.DataFrame({'filepath': filenames, 'label': labels_idx, 'label_tag': labels_name})


def create_dataset(data_dir="PlantVillage-Tomato"):
    # Setting up the directories
    train_dir = os.path.join(data_dir, 'Tomato', 'Train')
    valid_dir = os.path.join(data_dir, 'Tomato', 'Val')
    test_dir = os.path.join(data_dir, 'Tomato', 'Test')

    # Reading the filenames with their file paths
    train_files = glob.glob(os.path.join(train_dir, "*", "*"))
    valid_files = glob.glob(os.path.join(valid_dir, "*", "*"))
    test_files = glob.glob(os.path.join(test_dir, "*", "*"))

    # Reading the labels

    labels_dict = {}
    for filename in train_files:
        label = filename.split(os.path.sep)[3].split('___')[1]
        labels_dict[label] = labels_dict.get(label, 0) + 1

    labels = sorted(list(labels_dict.keys()))
    class_index_to_label_map = {}
    for i in range(len(labels)):
        class_index_to_label_map[int(i)] = labels[i]
    with open(os.path.join(data_dir, "class_mapping.json"), "w") as outfile:
        json.dump(class_index_to_label_map, outfile)

    print(f'[INFO] Total Classes Found: {len(labels_dict.keys())}')
    for k, v in labels_dict.items():
        print("\t", k, ": ", v)

    train_df = get_dataframe(train_files, labels_dict)
    valid_df = get_dataframe(valid_files, labels_dict)
    test_df = get_dataframe(test_files, labels_dict)

    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(data_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)


if __name__ == "__main__":
    create_dataset()
