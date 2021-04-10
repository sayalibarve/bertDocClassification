import torch
from mainfile import tokenizer, device
from pytorch_pretrained_bert import BertModel

def get_tokenized_text(text):
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    #print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    return tokenized_text, indexed_tokens, segments_ids

def get_encoded_layers(indexed_tokens, segments_ids):
    # Convert inputs to PyTorch tensors
    #tokens_tensor = torch.tensor([indexed_tokens])
    #segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    torch.cuda.empty_cache()
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(indexed_tokens, segments_ids)
    return encoded_layers

def get_embeddings(encoded_layers, emb_matrix, indexed_tokens, dimension = 768):
    token_embeddings = torch.stack(encoded_layers, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    #embedding_dict = {}

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token, word in zip(token_embeddings,indexed_tokens):

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        sum_vec = sum_vec.type(torch.DoubleTensor).to(device)
        # Use `sum_vec` to represent `token`.
        emb_matrix[word] = sum_vec[:dimension]

    return emb_matrix.cpu()
