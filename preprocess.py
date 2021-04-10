import pandas as pd
import re
import nltk
import numpy as np

nltk.download('punkt')
from nltk import tokenize
from tensorflow.keras.utils import to_categorical

def clean_str(string):
    try:
        string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string) 
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", "", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        string = re.sub(r"sssss", "", string)
        string = re.sub(r"-lrb-","",string)
        string = re.sub(r"-rrb-","",string)
    except Exception as e:
        print(e)
    return string.strip().lower()

def process_data(path):
    data_train=pd.read_csv(path, sep='\t', header=None, usecols=[4,6], names=["label","review"])
    reviews = []
    labels = []
    for idx in range(int(len(data_train["review"]))):
    #for idx in range(50):
        raw_text = data_train.review[idx]
        if type(raw_text) == str and raw_text != '' and raw_text is not None:
            reviews.append(tokenize.sent_tokenize(clean_str(raw_text)))
            labels.append(int(data_train.label[idx]))
    labels = to_categorical(np.asarray(labels))
    return reviews, labels
