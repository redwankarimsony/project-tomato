import os
import glob
import json
import shutil
import zipfile
import requests
import tqdm
from clint.textui import progress
import pandas as pd


def downloadFile(datasetUrl, fileName, chunksize=4096):
    """
    This function downloads a file from the given url and saves it in the given filename
    """
    try:
        response = requests.get(datasetUrl, stream=True)
        total_length = int(response.headers.get('content-length'))
        with open("PlantVillage-Tomato.zip", "wb") as f:
            for chunk in progress.bar(response.iter_content(chunk_size=chunksize),
                                      expected_size=(total_length / chunksize) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
    except Exception as exp:
        print("Problem Downloading\nError Log:\n" + exp)


def create_dataset(data_dir="PlantVillage-Tomato"):
    # Setting up the directories
    train_dir = os.path.join(data_dir, 'Train')
    valid_dir = os.path.join(data_dir, 'Val')
    test_dir = os.path.join(data_dir, 'Test')

    # Reading the filenames with their file paths
    train_files = glob.glob(os.path.join(train_dir, "*", "*"))
    valid_files = glob.glob(os.path.join(valid_dir, "*", "*"))
    test_files = glob.glob(os.path.join(test_dir, "*", "*"))

    print(f"Train Files: {len(train_files)}")
    print(f"Validation Files: {len(valid_files)}")
    print(f"Test Files: {len(test_files)}")

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


def deleteUnusedData(extractDir="Plant_leave_diseases_dataset_without_augmentation"):
    # Managing files
    allClasses = os.listdir(extractDir)

    # Deleting the unused class folder
    for classFolder in allClasses:
        if not classFolder.startswith("Tomato_"):
            try:
                # shutil.rmtree(os.path.join(extractDir, classFolder))
                print(os.path.join(extractDir, classFolder), "removed")
                allClasses.remove(classFolder)
            except Exception as exp:
                print(exp)


def scanOldDataset(dataDir="PlantVillage-Tomato"):
    trainDir = os.path.join(dataDir, "Train/")
    validDir = os.path.join(dataDir, "Val/")
    testDir = os.path.join(dataDir, "Test/")

    trainFiles = glob.glob(os.path.join(trainDir, "*", "*"))
    validFiles = glob.glob(os.path.join(validDir, "*", "*"))
    testFiles = glob.glob(os.path.join(testDir, "*", "*"))

    print(len(trainFiles), len(validFiles), len(testFiles))

    trainFiles = [x.replace(trainDir, "") for x in trainFiles]
    validFiles = [x.replace(validDir, "") for x in validFiles]
    testFiles = [x.replace(testDir, "") for x in testFiles]

    # Writing file lists in text file
    os.makedirs("splits", exist_ok=True)
    for split, files in zip(["train", 'valid', 'test'], [trainFiles, validFiles, testFiles]):
        try:
            with open(os.path.join("splits", f"{split}.txt"), "w") as f:
                f.write("\n".join(files))
        except Exception as exp:
            print(exp)


def extractZip(filename, targetDir=None):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        if targetDir:
            zip_ref.extractall(targetDir)
        else:
            zip_ref.extractall(".")


def arrangeDataset(rawDataDir="Plant_leave_diseases_dataset_without_augmentation", newDataDir="PlantVillage-Tomato"):
    trainFiles = "splits/train.txt"
    validFiles = "splits/valid.txt"
    testFiles = "splits/test.txt"

    try:
        shutil.rmtree(newDataDir)
    except Exception as exp:
        print(exp)

    os.makedirs(newDataDir, exist_ok=True)
    os.makedirs(os.path.join(newDataDir, "All-Tomato"), exist_ok=True)
    os.makedirs(os.path.join(newDataDir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(newDataDir, "Val"), exist_ok=True)
    os.makedirs(os.path.join(newDataDir, "Test"), exist_ok=True)

    imageFolders = os.listdir(rawDataDir)

    for imageFolder in imageFolders:
        if imageFolder.startswith("Tomato__"):
            print(imageFolder)
            shutil.copytree(os.path.join(rawDataDir, imageFolder), os.path.join(newDataDir, "All-Tomato", imageFolder))
            for split in ["Train", "Val", "Test"]:
                os.makedirs(os.path.join(newDataDir, split, imageFolder), exist_ok=True)

    with open(trainFiles, "r") as f:
        trainFiles = f.readlines()
    with open(testFiles, "r") as f:
        testFiles = f.readlines()
    with open(validFiles, "r") as f:
        validFiles = f.readlines()

    print("\n\nCopying Train Files...")
    for trainFile in tqdm.tqdm(trainFiles):
        src = os.path.join(newDataDir, "All-Tomato", trainFile.strip())
        dst = src.replace("All-Tomato", "Train")
        shutil.copy(src, dst)

    print("\n\nCopying Valid Files...")
    for validFile in tqdm.tqdm(validFiles):
        src = os.path.join(newDataDir, "All-Tomato", validFile.strip())
        dst = src.replace("All-Tomato", "Val")
        shutil.copy(src, dst)

    print("\n\nCopying Test Files...")
    for testFile in tqdm.tqdm(testFiles):
        src = os.path.join(newDataDir, "All-Tomato", testFile.strip())
        dst = src.replace("All-Tomato", "Test")
        shutil.copy(src, dst)

    # Copying the split files for use later on.
    try:
        shutil.copy(src=os.path.join("splits", "train.txt"),
                    dst=os.path.join(newDataDir, "train.txt"))
        shutil.copy(src=os.path.join("splits", "valid.txt"),
                    dst=os.path.join(newDataDir, "valid.txt"))
        shutil.copy(src=os.path.join("splits", "test.txt"),
                    dst=os.path.join(newDataDir, "test.txt"))
    except Exception as exp:
        print(exp)


def deleteUnwanteFiles():
    try:
        shutil.rmtree("Plant_leave_diseases_dataset_without_augmentation")
        print("[INFO:] Removing the raw data extraction folder")
    except Exception as exp:
        print(exp)

    try:
        os.remove("PlantVillage-Tomato.zip")
        print("[INFO:] Removing the raw zipped data file. ")
    except Exception as exp:
        print(exp)



if __name__ == "__main__":
    # Downloading Files
    datasetUrl = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    fileName = "PlantVillage-Tomato.zip"
    appDir = os.path.realpath(os.path.dirname(__file__))
    fileName = os.path.join(appDir, fileName)
    downloadFile(datasetUrl= datasetUrl, fileName=fileName)

    # Extracting Files
    extractZip(filename="PlantVillage-Tomato.zip")

    ## Create the splits from experiments
    # scanOldDataset()

    ## Taking out only the Tomato Image files for the project
    # arrangeDataset()

    # Deleting the unwanted files for the project.
    deleteUnwanteFiles()