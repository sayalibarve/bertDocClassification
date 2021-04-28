import HAN
import preprocess as pp
import bertprocess as bp
import config
import os
import torch
import numpy as np
import datetime
import math
import tensorflow as tf
import logging

from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, CSVLogger

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device properties : {torch.cuda.get_device_properties(device)}")
else:
    print("There is no GPU available. Running on CPU")
    device = torch.device("cpu")
    
#this creates embedding matrix which later used by HAN model. Set create_emb=True to create embeddings.
def get_inputs_labels_embedding_matrix(dataPath, create_emb):
    reviews,labels = pp.process_data(dataPath)
    tokens = []
    indexes = []
    segments = []
    #create review_input to pass to HAN model
    review_input = np.zeros((len(reviews), config.MAX_SENTS, config.MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            new_indexed_tokens = []
            if j < config.MAX_SENTS:
                tokenized_text, indexed_tokens, segments_ids = bp.get_tokenized_text(sent)
                tokens += tokenized_text
                indexes += indexed_tokens
                segments += segments_ids
                if len(indexed_tokens) < config.MAX_SENT_LENGTH:
                    new_indexed_tokens += [0] * (config.MAX_SENT_LENGTH - len(indexed_tokens))
                    review_input[i][j] = indexed_tokens + new_indexed_tokens
                else:
                    review_input[i][j] = indexed_tokens[:config.MAX_SENT_LENGTH]
    
    if create_emb==True:
        del reviews
        indexes = torch.tensor(indexes)
        segments = torch.tensor(segments)
        #create bert dataset 
        bert_dataset = TensorDataset(indexes, segments)
        batch_size = 512
        train_dataloader = DataLoader(bert_dataset, batch_size = batch_size, drop_last=True, num_workers=4)
        #initialize emb_matrix with random values
        emb_matrix = np.random.random((len(tokenizer.vocab) + 1, config.embedding_dim))
        for i, batch in enumerate(tqdm(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            emb_matrix = bp.get_embeddings(bp.get_encoded_layers(b_input_ids.view((1,len(b_input_ids))), b_input_mask.view((1,len(b_input_mask)))), emb_matrix, b_input_ids)
            if i % 1000 == 0:
                 np.save('embeddings_2013.npy',emb_matrix)
        #save embeddings
        np.save('embeddings_2013.npy',emb_matrix)
        emb_matrix = np.load("embeddings_2013.npy")
    else:
        emb_matrix = np.load("embedding.npy")

    return review_input, labels, emb_matrix
#this function is a provision for manipulation of learning rate
def scheduler(epoch, lr):
    if (epoch+1) % 5 == 0:
        return lr * 0.1
    else:
        return lr

if __name__=='__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(config.save_dir, "logfile.out"),
                        format='%(asctime)s %(message)s')
    #csv logger for accuracies
    csv_logger = CSVLogger(os.path.join(config.save_dir,'training.log'), append = True)
    if config.is_training == True:
        os.makedirs(os.path.join(config.save_dir, "checkpoints"), exist_ok=True)
        #save model checkpoints
        save_best_model = ModelCheckpoint(filepath=os.path.join(config.save_dir, "checkpoints", "cp-{epoch:04d}.ckpt"), \
                                        verbose=1, save_best_only = False, save_weights_only = True)
        
        os.makedirs(os.path.join(config.save_dir, "tensorboard"), exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=os.path.join(config.save_dir, "tensorboard"), histogram_freq=1)
        #lr_callback = LearningRateScheduler(scheduler)
        #get your processed data here
        trainReviews, trainLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.trainDataPath, create_emb=config.create_emb)
        testReviews, testLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.testDataPath, create_emb=False)
        valReviews, valLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.valDataPath, create_emb=False)
    
        #model and optimizer defined
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.lr)
        model = HAN.create_model(len(tokenizer.vocab) +1, config.embedding_dim, emb_matrix[:,:config.embedding_dim] )
        
        #set checkpoint path in config.py to load latest checkpoint
        if config.resume is not None:
            assert config.resume, "path not defined"
            #checkpoint_dir = os.path.dirname(config.resume)
            latest = tf.train.latest_checkpoint(config.resume)
            model.load_weights(latest)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            model.fit(trainReviews ,trainLabels, validation_data=(valReviews,valLabels), epochs=config.epoch,shuffle=True, batch_size=config.batch_size,callbacks=[tensorboard_callback,save_best_model, csv_logger])
            loss, acc = model.evaluate(testReviews,testLabels, callbacks=[csv_logger])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            model.fit(trainReviews , trainLabels, validation_data=(valReviews,valLabels), epochs=config.epoch,shuffle=True, batch_size=config.batch_size,callbacks=[tensorboard_callback,save_best_model, csv_logger])
            score = model.evaluate(testReviews, testLabels, batch_size=config.batch_size,callbacks=[csv_logger])
    else:
        testReviews, testLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.testDataPath, create_emb=False)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.lr)
        model = HAN.create_model(len(tokenizer.vocab) +1, config.embedding_dim, emb_matrix[:,:config.embedding_dim] )
        latest = tf.train.latest_checkpoint(config.resume)
        assert config.resume, "path not defined"
        model.load_weights(latest)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        loss, acc = model.evaluate(testReviews,testLabels, callbacks=[csv_logger])
