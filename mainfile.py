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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device properties : {torch.cuda.get_device_properties(device)}")
else:
    print("There is no GPU available. Running on CPU")
    device = torch.device("cpu")
    

def get_inputs_labels_embedding_matrix(dataPath, emb_flag):
    reviews,labels = pp.process_data(dataPath)
    tokens = []
    indexes = []
    segments = []

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
    
    if emb_flag==1:
        del reviews
        indexes = torch.tensor(indexes)
        segments = torch.tensor(segments)

        bert_dataset = TensorDataset(indexes, segments)
        batch_size = 512
        train_dataloader = DataLoader(bert_dataset, batch_size = batch_size, drop_last=True)
        emb_matrix = torch.tensor(np.random.random((len(tokenizer.vocab) + 1, config.embedding_dim))).to(device)
        for batch in tqdm(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_input_ids = b_input_ids.view((1,len(b_input_ids))).to(device)
            b_input_mask = b_input_mask.view((1,len(b_input_mask))).to(device)
            emb_matrix = bp.get_embeddings(bp.get_encoded_layers(b_input_ids,b_input_mask), emb_matrix, b_input_ids).to(device)
        emb_matrix = emb_matrix.cpu()
        emb_matrix = emb_matrix.numpy()[:,:]
        np.save('embeddings.npy',emb_matrix)
    else:
        emb_matrix = np.load("./embeddings.npy")

    return review_input, labels, emb_matrix

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

    emb_flag= config.emb_flag
    is_model_ready = config.model_ready
    checkpoint_dir = os.path.dirname(config.resume)

    os.makedirs(os.path.join(config.save_dir, "checkpoints"), exist_ok=True)
    save_best_model = ModelCheckpoint(filepath=os.path.join(config.save_dir, "checkpoints", "cp-{epoch:04d}.ckpt"), \
                                    verbose=1, save_best_only = False, save_weights_only = True)
    
    os.makedirs(os.path.join(config.save_dir, "tensorboard"), exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=os.path.join(config.save_dir, "tensorboard"), histogram_freq=1)
    #lr_callback = LearningRateScheduler(scheduler)

    trainReviews, trainLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.trainDataPath, emb_flag=emb_flag)
    testReviews, testLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.testDataPath, emb_flag=0)
    valReviews, valLabels, emb_matrix = get_inputs_labels_embedding_matrix(config.valDataPath, emb_flag=0)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.lr)
    model = HAN.create_model(len(tokenizer.vocab) +1, config.embedding_dim, emb_matrix )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
    model.summary()
    model.fit(trainReviews , trainLabels, validation_data=(valReviews,valLabels), epochs=config.epoch, batch_size=config.batch_size,callbacks=[tensorboard_callback,save_best_model])
    score = model.evaluate(testReviews, testLabels, batch_size=config.batch_size)


    if is_model_ready == 1:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(trainReviews , trainLabels, validation_data=(valReviews,valLabels), epochs=config.epoch, batch_size=config.batch_size,callbacks=[tensorboard_callback,save_best_model])
        loss, acc = model.evaluate(testReviews,testLabels)