import HAN
import preprocess as pp
import bertprocess as bp
import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("There are %d GPU(s) available." %torch.cuda.device_count())
  print("Device name:", torch.cuda.get_device_name(0))
  print(torch.cuda.get_device_properties(device))
else:
  device = torch.device("cpu")

if __name__=='__main__':
    data_path = "../data/yelp-2013-train.txt.ss"
    reviews, labels = pp.process_data(data_path)
    embedding_dim = 768
    MAX_SENT_LENGTH = 128   # Max number of words per sentence
    MAX_SENTS = 12  # Max number of sentences per document
    MAX_NB_WORDS = 20000    # Max number of words in tot
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = []
    indexes = []
    segments = []

    review_input = np.zeros((len(reviews), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            new_indexed_tokens = []
            if j < MAX_SENTS:
                tokenized_text, indexed_tokens, segments_ids = bp.get_tokenized_text(sent)
                tokens += tokenized_text
                indexes += indexed_tokens
                segments += segments_ids
                if len(indexed_tokens) < 128:
                    new_indexed_tokens += [0] * (MAX_SENT_LENGTH - len(indexed_tokens))
                    review_input[i][j] = indexed_tokens + new_indexed_tokens
                else:
                    review_input[i][j] = indexed_tokens[:128]
    indexes_tensor = torch.tensor(indexes)
    segment_tensor = torch.tensor(segments)

    bert_dataset = TensorDataset(indexes_tensor, segment_tensor)
    batch_size = 128
    train_dataloader = DataLoader(bert_dataset, batch_size = batch_size, drop_last=True)

    emb_matrix = torch.tensor(np.random.random((len(tokenizer.vocab) + 1, embedding_dim))).to(device)
    
    for idx, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_ids_l = b_input_ids.view((1,len(b_input_ids)))
        b_input_mask_l = b_input_mask.view((1,len(b_input_mask)))
        encoded_layers = bp.get_encoded_layers(b_input_ids_l, b_input_mask_l)
        emb_matrix = bp.get_embeddings(encoded_layers, emb_matrix, b_input_ids)

    emb_matrix = emb_matrix.cpu()
    emb_matrix = emb_matrix.numpy()[:,:]

    print("Embedding matrix size: ",emb_matrix.shape)

    model = HAN.create_model(len(tokenizer.vocab) +1, embedding_dim, emb_matrix )
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.summary()
    model.fit(review_input , labels, epochs=5, batch_size=100)

