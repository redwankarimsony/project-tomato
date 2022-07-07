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

