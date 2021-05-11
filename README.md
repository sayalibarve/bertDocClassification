
## Description:
This project experiments with addition of bert embeddings to Hierarchical Attention Network Model.

### Dataset Used: 
Yelp dataset is a text dataset that contains reviews which are rated from 1 to 5 (labels).  

## Software Requirements:
* python - 3.7
* torch - 1.8.1
* tensorflow - 2.0.0
* pytorch-pretrained-bert

## Hardware Requirements:
Nvidia GPU and cuda.

## Setup:
* Create a conda environment.
* Clone the repository.
* cd bertDocClassification.
* run the command to install necessary libraries : pip install -r requirements.txt
* make a folder with name "data".
* Add data to this data folder.
* The link for the data : https://drive.google.com/drive/folders/1D9SqCfkYrtGjjVuoorQTqx2II4bnUTWg?usp=sharing
* Download checkpoint from link : https://drive.google.com/drive/folders/1IH1JPCNGNohO5eH68KfXlq7ayj8SMZgi?usp=sharing
* Update the resume paramter in config.py with the path of the checkpoint folder(resume should be the path to the folder and not the .ckpt file).
* Download embedding.npy from link : https://drive.google.com/file/d/1CDyXat0B1FCSA2QeJbyQHaA7_LxZJmMS/view?usp=sharing
```config.py``` containes the required parameter values that can be configured before training or validation. Please check the file for more information about what parameters can be set. 

## Training:

```
./train.sh
```
## Validation:
```
./val.sh
```
