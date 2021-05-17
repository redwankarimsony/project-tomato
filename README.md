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
``` 
!git clone https://github.com/redwankarimsony/project-tomato.git
!cd project-tomato
!python train.py
```
### Evaluation
```
!python evaluate.py
```

