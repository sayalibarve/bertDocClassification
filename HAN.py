import config
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding, GRU, Bidirectional,TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))

        self._trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''
Model: Hierarchical Attention Neural Network
'''
def create_model(vocab_len, embedding_dim, emb_matrix):
    sentence_input = Input(shape=(config.MAX_SENT_LENGTH), dtype=tf.int32)
    embedding_layer = Embedding(vocab_len,
                                embedding_dim,
                                weights=[emb_matrix],
                                mask_zero=False,
                                input_length=config.MAX_SENT_LENGTH,
                                trainable=True)
    embedding = embedding_layer(sentence_input)
    
    l_lstm = Bidirectional(GRU(100, return_sequences=True,))(embedding)
    l_att = AttLayer(100)(l_lstm)
    sent_encoder = Model(inputs=[sentence_input], outputs=[l_att])
  
    review_input = Input(shape=(config.MAX_SENTS, config.MAX_SENT_LENGTH), dtype=tf.int32)
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(6, activation='softmax')(l_att_sent)

    model = Model(inputs=[review_input], outputs = [preds])
    return model
