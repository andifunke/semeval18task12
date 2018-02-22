"""
Neural models - new approach with sandwich design
"""
import numpy as np
from theano.scalar import float32
import keras
from keras.engine import Input, Model, Layer
from keras.layers import concatenate, Lambda, Dense, Dropout, Embedding, LSTM, Bidirectional, multiply, add, RNN, \
    SimpleRNN, Concatenate, Flatten, Convolution1D, ConvLSTM2D, TimeDistributed, Conv2D, MaxPooling2D, Conv1D, \
    MaxPooling1D, Multiply, Add, Average, Maximum, Dot, LeakyReLU, PReLU, ELU, ThresholdedReLU, Reshape
from keras import backend as K
from deprecated.attention_lstm import attention_3d_block
from constants import *


# 'cheating' metric - not working as expected
def dev_pred(y_true, y_pred):
    return K.cast(0.5, 'float32')


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
    weights = None if embeddings is None else [embeddings]
    el = list()
    for input_layer in input_layers:
        el.append(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=weights,
                            mask_zero=masking)(input_layer))
    return el


def get_bidi_lstm_layers(embedded_layers, names, lstm_size, activation=None, dropout=0.0):
    # bidi layers (bl)
    args = dict()
    if activation is not None:
        args['activation'] = activation

    bl = list()
    for embedded_layer, name in zip(embedded_layers, names):
        bl.append(Bidirectional(LSTM(lstm_size, return_sequences=True,
                                     dropout=dropout, recurrent_dropout=dropout,
                                     **args), name='BiDiLSTM_{}'.format(name))(embedded_layer))
    return bl


def get_activation_layers(layers, activation=PReLU):
    al = list()
    for layer in layers:
        al.append(activation()(layer))
    return al


def get_cnn_lstm_layers(embedded_layers, names, filters, kernel_size=3, dropout=0.0):
    # cnn lstm layers (cll)
    cll = list()
    for embedded_layer, name in zip(embedded_layers, names):
        cll.append(ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                              dropout=dropout, recurrent_dropout=dropout,
                              name='CNNLSTM_{}'.format(name))(embedded_layer))
    return cll


def get_complex_cnn_layers(embedded_layers, names, filters, activation=None):
    # nach: https://codekansas.github.io/blog/2016/language.html
    args = dict()
    if activation is not None:
        args['activation'] = activation
    cnns = [Conv1D(padding="same", kernel_size=filt, filters=filters,
                   name='conv1d_size{:d}'.format(filt), **args) for filt in [2, 3, 5, 7]]
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
        attention_vector_for_w0 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[0], bl[4], bl[5]]))
        attention_vector_for_w1 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[1], bl[4], bl[5]]))
    else:
        attention_vector_for_w0 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[0]]))
        attention_vector_for_w1 = max_pool_lambda_layer(concatenate([bl[2], bl[3], bl[1]]))

    return [attention_vector_for_w0, attention_vector_for_w1]


def get_model(options: dict, indices_to_vectors: dict):
    classifier = options['classifier']
    if classifier == 'LSTM_01':
        return get_baseline_model(options, indices_to_vectors)

    # converting embeddings to numpy 2d array: shape = (vocabulary_size, emb_dim)
    dropout = options['dropout']
    lstm_size = options['lstm_size']
    optimizer = options['optimizer']
    loss = options['loss']
    activation1 = options['activation1']
    activation2 = options['activation2']
    dense_factor = options.get('dense_factor', 1)
    padding = options['padding']
    rich_context = options['rich']
    filters = options.get('filters', lstm_size)
    use_input_layers = options.get('input_layers', [0, 1, 2, 3, 4, 5])

    embeddings = embedding_to_ndarray(indices_to_vectors)
    vocabulary = embeddings.shape[0]
    dimensionality = embeddings.shape[1]
    print('embeddings.shape', embeddings.shape)

    # by giving a different set of input layer indexes it is possible to alter the model
    # take care to train the model on related data sources
    names_default = CONTENT
    names = []
    for input_layer in use_input_layers:
        names.append(names_default[input_layer])

    # --- MODELS --------------------------------------------------------------------

    # LSTM_00 - most basic
    if classifier == 'LSTM_00':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        output_input = LSTM(lstm_size)(Concatenate()(bl[:2]))

    # based on LSTM_01 - only warrant 0 and 1 are used!
    elif classifier == 'LSTM_02':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[0])
        attention_warrant1 = LSTM(lstm_size)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # based on LSTM_02 - but more dropout
    elif classifier == 'LSTM_02a':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size, dropout=0.2)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size, dropout=0.2)(bl[0])
        attention_warrant1 = LSTM(lstm_size, dropout=0.2)(bl[1])
        conc = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(0.2, name='dropout')(conc)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # based on LSTM_02 - only claim and reason are used!
    elif classifier == 'LSTM_02b':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[2])
        attention_warrant1 = LSTM(lstm_size)(bl[3])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # combining 02 and 02b
    elif classifier == 'LSTM_02c':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        warrant0 = LSTM(lstm_size)(bl[0])
        warrant1 = LSTM(lstm_size)(bl[1])
        reason = LSTM(lstm_size)(bl[2])
        claim = LSTM(lstm_size)(bl[3])
        warrants = concatenate([warrant0, warrant1])
        reason_claim = concatenate([reason, claim])
        dropout_layer_w = Dropout(dropout, name='dropout_w')(warrants)
        dropout_layer_rc = Dropout(dropout, name='dropout_rc')(reason_claim)
        dense_w = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(dropout_layer_w)
        dense_rc = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_rc')(dropout_layer_rc)
        output_input = concatenate([dense_w, dense_rc])

    # variant of 02c using 16 subcategories
    elif classifier == 'LSTM_02d':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # only warrant layers used
        warrant0 = LSTM(lstm_size)(bl[0])
        warrant1 = LSTM(lstm_size)(bl[1])
        reason = LSTM(lstm_size)(bl[2])
        claim = LSTM(lstm_size)(bl[3])
        warrants = concatenate([warrant0, warrant1])
        reason_claim = concatenate([reason, claim])
        dropout_layer_w = Dropout(dropout, name='dropout_w')(warrants)
        dropout_layer_rc = Dropout(dropout, name='dropout_rc')(reason_claim)
        dense_w = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(dropout_layer_w)
        dense_rc = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_rc')(dropout_layer_rc)
        dense_main = Dense(16, activation=activation1, name='dense_main')(concatenate([dense_w, dense_rc]))
        output_input = dense_main

    # uses the max-pool layer: experimental
    elif classifier == 'LSTM_03':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        av = get_attention_vectors(bl, rich_context)
        attention_warrant0 = Dense(lstm_size, name='av0')(av[0])
        attention_warrant1 = Dense(lstm_size, name='av1')(av[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # LSTM_04 - very basic architecture using all input layers
    # slow and rather disappointing
    elif classifier == 'LSTM_04':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # dropout_input = LSTM(int(lstm_size * len(bl)))(Concatenate()(bl))
        dropout_input = Flatten()(NonMasking()(Concatenate()(bl)))
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # LSTM_05 - process reason and claim with each warrant independently
    # without suffix: multiply - no good results
    # add - ok, but nothing special
    # conc(atenate) - promising
    elif classifier[:7] == 'LSTM_05':
        mergers = dict(mul=Multiply, add=Add, avg=Average, max=Maximum, conc=Concatenate)
        Merger = mergers[classifier[7:]]
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # multiplying warrant0 and warrant1 separately with reason and claim - similar to attention
        warrant0 = LSTM(lstm_size)(Merger()([bl[0], bl[2], bl[3]]))
        warrant1 = LSTM(lstm_size)(Merger()([bl[1], bl[2], bl[3]]))
        dropout_layer = Dropout(dropout, name='dropout_w')(concatenate([warrant0, warrant1]))
        dense = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(dropout_layer)
        output_input = dense

    # LSTM_05 variant using Advanced Activation Function
    elif classifier[:7] == 'LSTM_15':
        mergers = dict(mul=Multiply, add=Add, avg=Average, max=Maximum, conc=Concatenate)
        Merger = mergers[classifier[7:]]
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size, activation='linear')
        al = get_activation_layers(bl, activation=PReLU)
        # multiplying warrant0 and warrant1 separately with reason and claim - similar to attention
        warrant0 = LSTM(lstm_size, activation='linear')(Merger()([bl[0], bl[2], bl[3]]))
        warrant1 = LSTM(lstm_size, activation='linear')(Merger()([bl[1], bl[2], bl[3]]))
        warrant0 = PReLU()(warrant0)
        warrant1 = PReLU()(warrant1)
        dropout_layer = Dropout(dropout, name='dropout_w')(concatenate([warrant0, warrant1]))
        dense = Dense(int(lstm_size * dense_factor), activation='linear', name='dense_w')(dropout_layer)
        dense = PReLU()(dense)
        output_input = dense

    # LSTM_06 - similar to LSTM_05, but concatenates claim and reson before merging with warrants and uses dot product
    elif classifier == 'LSTM_06':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        # multiplying warrant0 and warrant1 separately with reason and claim - similar to attention
        reason_claim = Concatenate()([bl[2], bl[3]])
        warrant0 = LSTM(lstm_size)(Dot(axes=(1, 1))([bl[0], reason_claim]))
        warrant1 = LSTM(lstm_size)(Dot(axes=(1, 1))([bl[1], reason_claim]))
        dropout_layer = Dropout(dropout, name='dropout_w')(concatenate([warrant0, warrant1]))
        dense = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(dropout_layer)
        output_input = dense

    # like LSTM_05 - but using cnn instead of lstm
    elif classifier[:11] == 'LSTM_CNN_07':
        mergers = dict(mul=Multiply, add=Add, avg=Average, max=Maximum, con=Concatenate)
        Merger = mergers.get(classifier[11:], Add)
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        warrant0 = Conv1D(lstm_size, (5,), activation='relu')(Merger()([bl[0], bl[2], bl[3]]))
        warrant1 = Conv1D(lstm_size, (5,), activation='relu')(Merger()([bl[1], bl[2], bl[3]]))
        maxpol0 = MaxPooling1D()(warrant0)
        maxpol1 = MaxPooling1D()(warrant1)
        dropout_layer = Dropout(dropout, name='dropout_w')(Concatenate()([maxpol0, maxpol1]))
        dense = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(Flatten()(dropout_layer))
        output_input = dense

    # like LSTM_05conc, but concatenate on different axis
    # (which makes more sense in theory but doesn't work in real life)
    elif classifier == 'LSTM_08':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        bl = get_bidi_lstm_layers(el, names, lstm_size)
        warrant0 = LSTM(lstm_size)(Concatenate(axis=1)([bl[0], bl[2], bl[3]]))
        warrant1 = LSTM(lstm_size)(Concatenate(axis=1)([bl[1], bl[2], bl[3]]))
        dropout_layer = Dropout(dropout, name='dropout_w')(concatenate([warrant0, warrant1]))
        dense = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_w')(dropout_layer)
        output_input = dense


    # ##############################################################################
    # CNN

    # CNN_LSTM_02 - using multiple (parallel) CNN filters before the LSTM. only warrant 0 and 1 are used!
    elif classifier == 'CNN_LSTM_02':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        cl = get_complex_cnn_layers(el, names, filters)
        bl = get_bidi_lstm_layers(cl, names, lstm_size)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size)(bl[0])
        attention_warrant1 = LSTM(lstm_size)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # CNN_LSTM_02b - like CNN_LSTM_02, but uses custom activation function on all layers (reLu by default)
    # more efficient, but bad results
    elif classifier == 'CNN_LSTM_02b':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        cl = get_complex_cnn_layers(el, names, filters, activation=activation1)
        bl = get_bidi_lstm_layers(cl, names, lstm_size, activation=activation1)
        # only warrant layers used
        attention_warrant0 = LSTM(lstm_size, activation=activation1)(bl[0])
        attention_warrant1 = LSTM(lstm_size, activation=activation1)(bl[1])
        dropout_input = concatenate([attention_warrant0, attention_warrant1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # using ConvLSTM2D layer and all input layers short of debate - not working yet because of input shape (5d)
    elif classifier == 'CNN_LSTM_03':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        cll = get_cnn_lstm_layers(el, names, filters)
        dropout_input = concatenate(cll[:-1])
        dropout_layer = Dropout(dropout, name='dropout')(dropout_input)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # using ConvLSTM2D layer, like CNN_LSTM_03, but more simple without additional layers - also not working
    elif classifier == 'CNN_LSTM_04':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
        cll = get_cnn_lstm_layers(el, names, filters, dropout=dropout)
        output_input = concatenate(cll[:-1])

    # just a sketch for serial CNN filters with max-pooling - not really working
    elif classifier == 'CNN_LSTM_05':
        il = get_input_layers(names, padding)
        print(il[0].shape)
        el = embed_inputs(il, embeddings, padding, masking=False)
        used_layers = el[:-1]
        x = Concatenate(input_shape=(None, padding, dimensionality * len(used_layers)))(used_layers)
        y = Conv1D(int(filters/2), 7, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = Conv1D(int(filters/4), 5, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = Conv1D(int(filters/8), 3, activation='relu', padding='same')(x)
        x = MaxPooling1D()(y)
        y = LSTM(lstm_size)(x)
        dropout_layer = Dropout(dropout, name='dropout')(y)
        dense1 = Dense(int(lstm_size * dense_factor), activation=activation1, name='dense_main')(dropout_layer)
        output_input = dense1

    # using ConvLSTM2D layer, like CNN_LSTM_03, but more simple without additional layers - also not working
    elif classifier == 'CNN_01':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        cnn = get_complex_cnn_layers(el, names, filters)
        conc = concatenate(cnn[:2])
        output_input = Bidirectional(LSTM(lstm_size, dropout=0.2))(conc)

    # using ConvLSTM2D layer, like CNN_LSTM_03, but more simple without additional layers - also not working
    elif classifier == 'CNN_02':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:2])
        cnn = Conv1D(padding, (5,), activation='relu')(conc)
        output_input = Bidirectional(LSTM(lstm_size, dropout=0.2))(cnn)

    # using ConvLSTM2D layer, like CNN_LSTM_03, but more simple without additional layers - also not working
    elif classifier == 'CNN_03':
        il = get_input_layers(names, padding)
        print(il[0].shape)
        el = embed_inputs(il, embeddings, padding, masking=False)
        print(el[0].shape)
        layers = el[:2]
        conc = Concatenate(axis=1)(layers)
        print(conc.shape)
        dim4 = Reshape((vocabulary, padding, dimensionality, len(layers)))(conc)
        print(dim4.shape)
        cnn = Conv2D(padding, (5, 5), activation=PReLU())(dim4)
        print(cnn.shape)
        output_input = Bidirectional(LSTM(lstm_size, dropout=0.2))(cnn)
        print(output_input.shape)

    # ##################################################################################
    # ATTENTION

    # based on ATT_LSTM_01 (https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_dense.py)
    # too random
    elif classifier == 'ATT_LSTM_02':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding)
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
        attention_mul = Dense(int(lstm_size * dense_factor))(attention_mul)
        dense1 = Dense(int(lstm_size * dense_factor * 0.5), activation=activation1)(attention_mul)
        output_input = dense1

    # Attention LSTM from https://github.com/philipperemy/keras-attention-mechanism
    # APPLY_ATTENTION_BEFORE_LSTM
    # working pretty good!
    elif classifier == 'ATT_LSTM_03':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:2])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        output_input = lstm

    # augmented with dropout and additional dense layer
    elif classifier == 'ATT_LSTM_03a2':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:2])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        dropout_layer = Dropout(dropout, name='dropout')(lstm)
        output_input = dropout_layer

    # !
    elif classifier == 'ATT_LSTM_03b':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:3])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        output_input = lstm

    # with dropout and dense
    elif classifier == 'ATT_LSTM_03b3':
        print('this one')
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:3])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        dropout_layer = Dropout(0.5, name='dropout')(lstm)
        output_input = dropout_layer

    elif classifier == 'ATT_LSTM_03c':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:2] + [el[3]])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        output_input = lstm

    elif classifier == 'ATT_LSTM_03d':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:4])
        attention_mul = attention_3d_block(conc, padding)
        lstm = LSTM(lstm_size, return_sequences=False)(attention_mul)
        output_input = lstm

    elif classifier == 'ATT_LSTM_03e':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()(el[:2])
        attention_mul = attention_3d_block(conc, padding)
        conc2 = Concatenate()([attention_mul] + el[2:4])
        lstm = LSTM(lstm_size, return_sequences=False)(conc2)
        output_input = lstm

    # Attention LSTM from https://github.com/philipperemy/keras-attention-mechanism
    # APPLY_ATTENTION_AFTER_LSTM
    elif classifier == 'ATT_LSTM_04':
        il = get_input_layers(names, padding)
        el = embed_inputs(il, embeddings, padding, masking=False)
        conc = Concatenate()([el[0], el[1]])
        lstm = LSTM(lstm_size, return_sequences=True)(conc)
        attention_mul = attention_3d_block(lstm, padding)
        output_input = Flatten()(attention_mul)

    else:
        raise ValueError('wrong classifier shortcut')

    output_layer = Dense(1, activation=activation2)(output_input)
    model = Model(inputs=il, outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', dev_pred])

    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='./figures/{}.png'.format(classifier))

    return model


def get_baseline_model(options: dict, indices_to_vectors: dict):
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, emb_dim)
    max_len = options.get('padding')
    lstm_size = options.get('lstm_size')
    dropout = options.get('dropout')
    optimizer = options.get('optimizer')
    loss = options.get('loss')
    activation1 = options.get('activation1')
    activation2 = options.get('activation2')

    embeddings = np.asarray([np.array(x, dtype=float32) for x in indices_to_vectors.values()])
    print('LSTM_01: embeddings.shape', embeddings.shape)

    # define basic four input layers - for warrant0, warrant1, reason, claim
    sequence_layer_warrant0_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_warrant0")
    sequence_layer_warrant1_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_warrant1")
    sequence_layer_reason_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_reason")
    sequence_layer_claim_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_claim")
    sequence_layer_debate_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_input_debate")

    # now define embedded layers of the input
    embedded_layer_warrant0_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_warrant0_input)
    embedded_layer_warrant1_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_warrant1_input)

    bidi_lstm_layer_warrant0 = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_W0')(embedded_layer_warrant0_input)
    bidi_lstm_layer_warrant1 = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_W1')(embedded_layer_warrant1_input)

    # max-pooling
    max_pool_lambda_layer = Lambda(lambda x: keras.backend.max(x, axis=1, keepdims=False),
                                   output_shape=lambda x: (x[0], x[2]))
    max_pool_lambda_layer.supports_masking = True

    attention_warrant0 = LSTM(lstm_size)(bidi_lstm_layer_warrant0)
    attention_warrant1 = LSTM(lstm_size)(bidi_lstm_layer_warrant1)

    # concatenate them
    dropout_layer = Dropout(dropout)(concatenate([attention_warrant0, attention_warrant1]))

    # and add one extra dense layer
    dense1 = Dense(int(lstm_size), activation=activation1)(dropout_layer)
    output_layer = Dense(1, activation=activation2)(dense1)

    model = Model(inputs=[sequence_layer_warrant0_input, sequence_layer_warrant1_input, sequence_layer_reason_input,
                          sequence_layer_claim_input, sequence_layer_debate_input], outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', dev_pred])

    return model
