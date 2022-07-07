This repository is the official implementation of the paper published in IEEE Access Titled [Less is More: Lighter and Faster Deep Neural Architecture for Tomato Leaf Disease Classification](https://ieeexplore.ieee.org/document/9810234).

This branch contains the inference code. 


## Steps To Run the Code
### Step 1: Install Anaconda
Go to the [Anaconda Website](https://www.anaconda.com/products/distribution) and choose a Python 3.x graphical installer (A) or a Python 2.x graphical installer (B). If you aren't sure which Python version you want to install, choose Python 3. Do not choose both.

### Step 2: Clone the Repository
In order to clone the repository, use the following git command in your command line. 
```
git clone https://github.com/redwankarimsony/project-tomato.git
```
and then move into the project directory with 
```
cd project-tomato
```
### Step 3: Create a Python Environment
The anaconda virtual environment used to run the code is already exported in the `.yml` files here. If you are using Linux, use the following code to create the environment. This will create a new environment named `tomato`
```
conda env create -f environment_linux.yml
```
Even you are using the other operating system, use the following command to create the anaconda environment. 
```
conda env create -f environment_all_os.yml
```
Activate the environment with the following command

```
conda activate tomato
```

### Step 4: Download the Dataset
In order to download the dataset, run the following script with 
```(Python)
python download_dataset.py
```
To run the inference code, run the follwing python file with the given command.
```(Python)
python inference.py
```








# Repository Structure:
```(Python)
├── config.json
├── dataset_preparation.py
├── dataset.py
├── download_dataset.py
├── evaluate.py
├── inference-config.json
├── inference.py
├── model.py
├── PlantVillage-Tomato
│   ├── All-Tomato
│   ├── Test
│   ├── test.csv
│   ├── test.txt
│   ├── Train
│   ├── train.csv
│   ├── train.txt
│   ├── trash.py
│   ├── Val
│   ├── valid.csv
│   └── valid.txt
├── README.md
├── saved_models
│   ├── MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5
│   └── MobileNetV2_WithCLAHE_NoAug_WithoutDense_ValBest.h5
├── splits
│   ├── test.txt
│   ├── train.txt
│   └── valid.txt
├── train.py
└── utils.py

```



# How to Run it
## Run on Google Colab
### Training 
You can use google colab to run it on the go. It will not need any installation of packages. Just to run it in different configuration, change the file `config.json`. After you are done training, the whole training log and different graphs like accuracy, loss and learning rate plot images are stored in the checkpoint folder. To know the location of the checkpoint folder, check the `checkpoint_filepath` attribute's in the `config.json  file.
Open a google colab notebook and run the following block of code to start training.
``` 
!git clone https://github.com/redwankarimsony/project-tomato.git
!cd project-tomato
!python train.py
```
### Evaluation
Once you are done training, you can evaluate your model on the test dataset. Check the checkpoint directory to have a look at the performance graphs like confusion matrix. The `misclassified` directory inside checkpoint directory contains the misclassified samples. 
```
!python evaluate.py
```

