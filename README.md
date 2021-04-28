
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

## Training:

```
./train.sh 
```
Clone the repository.
cd bertDocClassification
make a data folder.
Add data to this data folder.
The embedding file and checkpoints can be provided as needed.
```config.py``` containes the required parameter values that can be configured before training or validation. Please check the file for more information about what parameters can be set. 
