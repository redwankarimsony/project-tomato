# Repository Structure:
```(Python)
.
├── NoCLAHE-NoAug-NoDense
│   ├── graphs
│   │   ├── 1.accuracy-comparison.png
│   │   ├── 2.loss-comparison.png
│   │   ├── 3.learning-rate.png
│   │   └── 4.confusion-matrix.png
│   ├── model_snapshot.ckpt
│   │   ├── assets
│   │   ├── variables
│   │   └── saved_model.pb
│   ├── saved_model
│   │   ├── assets
│   │   ├── variables
│   │   └── saved_model.pb
│   └── train_log.csv
├── PlantVillage-Tomato
│   ├── Tomato
│   │   ├── Test
│   │   ├── Train
│   │   └── Val
│   ├── class_mapping.json
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── config.json
├── dataset_preparation.py
├── dataset.py
├── evaluate.py
├── filetree.txt
├── model.py
├── README.md
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

