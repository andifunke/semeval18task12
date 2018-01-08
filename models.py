"""
Neural models - new approach with sandwich design
"""
from keras import Sequential

from main import dev_pred
import keras
import numpy as np
from keras.engine import Input, Model, Layer
from keras.layers import concatenate, Lambda, Dense, Dropout, Embedding, LSTM, Bidirectional, multiply, add, RNN, \
    SimpleRNN, Concatenate, Flatten, Convolution1D, ConvLSTM2D, TimeDistributed, Conv2D, MaxPooling2D, Conv1D, \
    MaxPooling1D
from theano.scalar import float32
import models_vintage


SHORTCUTS = {'LSTM_02', 'LSTM_03', 'LSTM_04',
             'CNN_LSTM_02', 'CNN_LSTM_02b', 'CNN_LSTM_03', 'CNN_LSTM_04', 'CNN_LSTM_05',
             'ATT_LSTM_02',
             }
VINTAGE = dict(
    LSTM_01=models_vintage.get_lstm_intra_warrant,
    ATT_LSTM_01=models_vintage.get_attention_lstm_intra_warrant,
    CNN_LSTM_01=models_vintage.get_cnn_lstm,
)
SHORTCUTS.update(VINTAGE.keys())


# from: https://github.com/keras-team/keras/issues/4978
class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        # input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape


def get_shortcuts():
    return SHORTCUTS


def embedding_to_ndarray(word_embedding):
    return np.asarray([np.array(x, dtype=float32) for x in word_embedding.values()])


def get_input_layers(names, max_len):
    # define basic four input layers - for warrant0, warrant1, reason, claim, debate
    il = list()
    for name in names:
        il.append(Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_{}".format(name)))
    return il


def embed_inputs(input_layers, embeddings, max_len, masking=True):
    # now define embedded layers of the input
    # embedding layers (el)
    el = list()
    for input_layer in input_layers:
        el.append(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings],
                            mask_zero=masking)(input_layer))
    return el


def get_bidi_lstm_layers(embedded_layers, names, lstm_size, activation=None):
    # bidi layers (bl)
    args = dict()
    if activation is not None:
        args['activation'] = activation
    bl = list()
    for embedded_layer, name in zip(embedded_layers, names):
        bl.append(Bidirectional(LSTM(lstm_size, return_sequences=True, **args), name='BiDiLSTM_{}'.format(name))(embedded_layer))
    return bl


def get_cnn_lstm_layers(embedded_layers, names, filters, kernel_size=3, dropout=0.0):
    # cnn lstm layers (cll)
    cll = list()
    for embedded_layer, name in zip(embedded_layers, names):
        cll.append(ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                              dropout=dropout, recurrent_dropout=dropout,
                              name='CNNLSTM_{}'.format(name))(embedded_layer))
    return cll


def get_cnn_layer(embedded_layers, names, filters, activation=None):
    # nach: https://codekansas.github.io/blog/2016/language.html
    args = dict()
    if activation is not None:
        args['activation'] = activation
    cnns = [Convolution1D(padding="same", kernel_size=filt, filters=filters, name='conv1d_size{:d}'.format(filt), **args)
            for filt in [2, 3, 5, 7]]
    # cnn layers (cl)
    cl = list()
    for embedded_layer, name in zip(embedded_layers, names):
        cl.append(Concatenate(name='cnn_{}'.format(name))([cnn(embedded_layer) for cnn in cnns]))
    return cl


def get_attention_vectors(bidi_layers, rich_context=True):
    bl = bidi_layers
    # max-pooling
    max_pool_lambda_layer = Lambda(
        lambda x: keras.backend.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    max_pool_lambda_layer.supports_masking = True

    if rich_context:
        attention_vector_for_w0 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[0], bl[4]]))
        attention_vector_for_w1 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[1], bl[4]]))
    else:
        attention_vector_for_w0 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[0]]))
        attention_vector_for_w1 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[1]]))

    return [attention_vector_for_w0, attention_vector_for_w1]


def get_model(classifier, word_index_to_embeddings_map, max_len, rich_context, **kwargs):
    if classifier in VINTAGE:
        return VINTAGE[classifier](word_index_to_embeddings_map, max_len, rich_context, **kwargs)

    # converting embeddings to numpy 2d array: shape = (vocabulary_size, emb_dim)
    embeddings = embedding_to_ndarray(word_index_to_embeddings_map)
    print('embeddings.shape', embeddings.shape)

    # second empedding ?
    rich_embedding = False
    if 'embeddings2' in kwargs:
        word_index_to_embeddings_map2 = kwargs['embeddings2']
        embeddings2 = embedding_to_ndarray(word_index_to_embeddings_map2)
        rich_embedding = True

    dimensionality = embeddings.shape[1]
    lstm_size = kwargs.get('lstm_size')
    dropout = kwargs.get('dropout')
    optimizer = kwargs.get('optimizer')
    loss = kwargs.get('loss')
    activation1 = kwargs.get('activation1')
    activation2 = kwargs.get('activation2')
    factor = kwargs.get('dense_factor')
    filters = kwargs.get('filters', lstm_size)
    use_input_layers = kwargs.get('input_layers', [0, 1, 2, 3, 4])
    assert factor
    assert lstm_size
    assert dropout

    # by giving a different set of input layer indexes it is possible to alter the model
    # take care to train the model on related data sources
    # TODO: test for non unique values (=> duplicate layer names)
    names_default = ['warrant0', 'warrant1', 'reason', 'claim', 'debate']
    names = []
    for input_layer in use_input_layers:
        names.append(names_default[input_layer])

    # TODO: rich (i.e. second) embedding

    # original Attention model
    # from attention_lstm.attention_lstm import AttentionLSTM
    # difficult to reproduce since this AtenntionLSTM model is not compatible with Keras 2 anymore:
    # attention_warrant0 = AttentionLSTM(lstm_size, attention_vector_for_w0)(bidi_lstm_layer_warrant0)
    # attention_warrant1 = AttentionLSTM(lstm_size, attention_vector_for_w1)(bidi_lstm_layer_warrant1)

    # based on LSTM_01 - take care: only warrant 0 and 1 are used!
    if classifier == 'LSTM_02':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[0])
        attention_warrant1 = LSTM(lstm_size)(bl[1])
        # TODO: use 'add' instead of 'concatenate' to reconstruct possible bug in original implementation
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # uses the max-pool layer: experimental
    elif classifier == 'LSTM_03':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        av = get_attention_vectors(bl, rich_context)
        attention_warrant0 = Dense(lstm_size, name='av0')(av[0])
        attention_warrant1 = Dense(lstm_size, name='av1')(av[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # LSTM_04 - very basic architecture using all input layers
    # slow and rather disappointing
    elif classifier == 'LSTM_04':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # dropout_input = LSTM(int(lstm_size * len(bl)))(Concatenate()(bl))
        dropout_input = Flatten()(NonMasking()(Concatenate()(bl)))
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # CNN_LSTM_02 - using multiple (parallel) CNN filters before the LSTM. only warrant 0 and 1 are used!
    elif classifier == 'CNN_LSTM_02':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len, masking=False)
        cl = get_cnn_layer(el, names, filters)
        bl = get_bidi_lstm_layers(cl, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[0])
        attention_warrant1 = LSTM(lstm_size)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # CNN_LSTM_02b - like CNN_LSTM_02, but uses custom activation function on all layers (reLu by default)
    # more efficient, but bad results
    elif classifier == 'CNN_LSTM_02b':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len, masking=False)
        cl = get_cnn_layer(el, names, filters, activation=activation1)
        bl = get_bidi_lstm_layers(cl, names, lstm_size, activation=activation1)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size, activation=activation1)(bl[0])
        attention_warrant1 = LSTM(lstm_size, activation=activation1)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # using ConvLSTM2D layer and all input layers short of debate - not working yet because of input shape (5d)
    elif classifier == 'CNN_LSTM_03':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        cll = get_cnn_lstm_layers(el, names, filters)
        dropout_input = concatenate(cll[:-1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # using ConvLSTM2D layer, like CNN_LSTM_03, but more simple without additional layers - also not working
    elif classifier == 'CNN_LSTM_04':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        cll = get_cnn_lstm_layers(el, names, filters, dropout=dropout)
        output_input = concatenate(cll[:-1])

    # just a sketch for serial CNN filters with max-pooling - not really working
    elif classifier == 'CNN_LSTM_05':
        il = get_input_layers(names, max_len)
        print(il[0].shape)
        el = embed_inputs(il, embeddings, max_len, masking=False)
        used_layers = el[:-1]
        x = Concatenate(input_shape=(None, max_len, dimensionality * len(used_layers)))(used_layers)
        y = Conv1D(int(filters/2), 7, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = Conv1D(int(filters/4), 5, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = Conv1D(int(filters/8), 3, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = LSTM(lstm_size)(x)
        dropout_layer = Dropout(dropout, name='dropout')(y)
        dense1 = Dense(int(lstm_size * factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # based on ATT_LSTM_01 (https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_dense.py)
    # pure randomness - unusable
    elif classifier == 'ATT_LSTM_02':
        il = get_input_layers(names, max_len)
        el = embed_inputs(il, embeddings, max_len)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[0])
        attention_warrant1 = LSTM(lstm_size)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        # experimental
        attention_probs = Dense(lstm_size * 2, activation='softmax', name='attention_vec')(dropout_layer)
        attention_mul = multiply([dropout_layer, attention_probs], name='attention_mul')
        # ATTENTION PART FINISHES HERE
        attention_mul = Dense(int(lstm_size * factor))(attention_mul)
        dense1 = Dense(int(lstm_size * factor * 0.5), activation=activation1)(attention_mul)
        output_input = dense1

    # TODO:
    # Attention LSTM from https://github.com/philipperemy/keras-attention-mechanism
    # https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

    else:
        raise ValueError('wrong classifier shortcut')

    output_layer = Dense(1, activation=activation2)(output_input)
    model = Model(inputs=il, outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', dev_pred])

    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='./figures/{}.png'.format(classifier))

    return model
