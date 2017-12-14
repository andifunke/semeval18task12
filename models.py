"""
Neural models
"""

import keras
import numpy as np
from keras.engine import Input, Model
from keras.layers import concatenate, Lambda, Dense, Dropout, Embedding, LSTM, Bidirectional
from theano.scalar import float32
# from attention_lstm import AttentionLSTM


def get_attention_lstm(word_index_to_embeddings_map, max_len, rich_context: bool = False, **kwargs):
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.asarray([np.array(x, dtype=float32) for x in word_index_to_embeddings_map.values()])
    word_index_to_embeddings_map2 = kwargs['embeddings2']
    embeddings2 = np.asarray([np.array(x, dtype=float32) for x in word_index_to_embeddings_map2.values()])
    print('embeddings.shape', embeddings.shape)

    lstm_size = kwargs.get('lstm_size')
    dropout = kwargs.get('dropout')
    optimizer = kwargs.get('optimizer')
    loss = kwargs.get('loss')
    activation1 = kwargs.get('activation1')
    activation2 = kwargs.get('activation2')
    assert lstm_size
    assert dropout

    # define basic four input layers - for warrant0, warrant1, reason, claim
    sequence_layer_warrant0_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_warrant0_input")
    sequence_layer_warrant1_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_warrant1_input")
    sequence_layer_reason_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_reason_input")
    sequence_layer_claim_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_claim_input")
    sequence_layer_debate_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_debate_input")

    # now define embedded layers of the input
    embedded_layer_warrant0_input = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len,
                                              weights=[embeddings], mask_zero=True)(sequence_layer_warrant0_input)
    embedded_layer_warrant1_input = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len,
                                              weights=[embeddings], mask_zero=True)(sequence_layer_warrant1_input)
    embedded_layer_reason_input = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len,
                                            weights=[embeddings], mask_zero=True)(sequence_layer_reason_input)
    embedded_layer_claim_input = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len,
                                           weights=[embeddings], mask_zero=True)(sequence_layer_claim_input)
    embedded_layer_debate_input = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len,
                                            weights=[embeddings], mask_zero=True)(sequence_layer_debate_input)

    bidi_lstm_layer_reason = Bidirectional(LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Reason')(
        embedded_layer_reason_input)
    bidi_lstm_layer_claim = Bidirectional(LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Claim')(
        embedded_layer_claim_input)
    # add context to the attention layer
    bidi_lstm_layer_debate = Bidirectional(LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Context')(
        embedded_layer_debate_input)

    if rich_context:
        # merge reason and claim
        context_concat = concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_debate])
    else:
        context_concat = concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim])

    # max-pooling
    max_pool_lambda_layer = Lambda(lambda x: keras.backend.max(x, axis=1, keepdims=False),
                                   output_shape=lambda x: (x[0], x[2]))
    max_pool_lambda_layer.supports_masking = True
    attention_vector = max_pool_lambda_layer(context_concat)

    attention_warrant0 = LSTM(lstm_size, attention_vector)(embedded_layer_warrant0_input)
    attention_warrant1 = LSTM(lstm_size, attention_vector)(embedded_layer_warrant1_input)

    # concatenate them
    dropout_layer = Dropout(dropout)(concatenate([attention_warrant0, attention_warrant1]))

    # and add one extra layer with ReLU
    dense1 = Dense(int(lstm_size / 2), activation=activation1)(dropout_layer)
    output_layer = Dense(1, activation=activation2)(dense1)

    model = Model([sequence_layer_warrant0_input, sequence_layer_warrant1_input, sequence_layer_reason_input,
                   sequence_layer_claim_input, sequence_layer_debate_input], outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # from keras.utils.visualize_util import plot
    # plot(model, show_shapes=True, to_file='/tmp/model-att.png')

    # from keras.utils.visualize_util import plot
    # plot(model, show_shapes=True, to_file='/tmp/attlstm.png')

    return model


def get_attention_lstm_intra_warrant(word_index_to_embeddings_map, max_len, rich_context, **kwargs):
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.asarray([np.array(x, dtype=float32) for x in word_index_to_embeddings_map.values()])
    print('embeddings.shape', embeddings.shape)
    rich_embedding = False
    if 'embeddings2' in kwargs:
        word_index_to_embeddings_map2 = kwargs['embeddings2']
        embeddings2 = np.asarray([np.array(x, dtype=float32) for x in word_index_to_embeddings_map2.values()])
        rich_embedding = True

    lstm_size = kwargs.get('lstm_size')
    dropout = kwargs.get('dropout')
    optimizer = kwargs.get('optimizer')
    loss = kwargs.get('loss')
    activation1 = kwargs.get('activation1')
    activation2 = kwargs.get('activation2')
    assert lstm_size
    assert dropout

    # define basic four input layers - for warrant0, warrant1, reason, claim
    sequence_layer_warrant0_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_warrant0_input")
    sequence_layer_warrant1_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_warrant1_input")
    sequence_layer_reason_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_reason_input")
    sequence_layer_claim_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_claim_input")
    sequence_layer_debate_input = Input(shape=(max_len,), dtype='int32', name="sequence_layer_debate_input")

    # now define embedded layers of the input
    embedded_layer_warrant0_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_warrant0_input)
    embedded_layer_warrant1_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_warrant1_input)
    embedded_layer_reason_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_reason_input)
    embedded_layer_claim_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_claim_input)
    embedded_layer_debate_input = Embedding(
        embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings], mask_zero=True)(
        sequence_layer_debate_input)

    bidi_lstm_layer_warrant0 = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_W0')(embedded_layer_warrant0_input)
    bidi_lstm_layer_warrant1 = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_W1')(embedded_layer_warrant1_input)
    bidi_lstm_layer_reason = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_Reason')(embedded_layer_reason_input)
    bidi_lstm_layer_claim = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_Claim')(embedded_layer_claim_input)
    # add context to the attention layer
    bidi_lstm_layer_debate = Bidirectional(
        LSTM(lstm_size, return_sequences=True), name='BiDiLSTM_Context')(embedded_layer_debate_input)

    if rich_embedding:
        embedded_layer_warrant0_input2 = Embedding(
            embeddings2.shape[0], embeddings2.shape[1], input_length=max_len, weights=[embeddings2], mask_zero=True)(
            sequence_layer_warrant0_input)
        embedded_layer_warrant1_input2 = Embedding(
            embeddings2.shape[0], embeddings2.shape[1], input_length=max_len, weights=[embeddings2], mask_zero=True)(
            sequence_layer_warrant1_input)
        embedded_layer_reason_input2 = Embedding(
            embeddings2.shape[0], embeddings2.shape[1], input_length=max_len, weights=[embeddings2], mask_zero=True)(
            sequence_layer_reason_input)
        embedded_layer_claim_input2 = Embedding(
            embeddings2.shape[0], embeddings2.shape[1], input_length=max_len, weights=[embeddings2], mask_zero=True)(
            sequence_layer_claim_input)
        embedded_layer_debate_input2 = Embedding(
            embeddings2.shape[0], embeddings2.shape[1], input_length=max_len, weights=[embeddings2], mask_zero=True)(
            sequence_layer_debate_input)

        bidi_lstm_layer_warrant0_2 = Bidirectional(
            LSTM(lstm_size, return_sequences=True), name='BiDiLSTM W0')(embedded_layer_warrant0_input2)
        bidi_lstm_layer_warrant1_2 = Bidirectional(
            LSTM(lstm_size, return_sequences=True), name='BiDiLSTM W1')(embedded_layer_warrant1_input2)
        bidi_lstm_layer_reason_2 = Bidirectional(
            LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Reason')(embedded_layer_reason_input2)
        bidi_lstm_layer_claim_2 = Bidirectional(
            LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Claim')(embedded_layer_claim_input2)
        # add context to the attention layer
        bidi_lstm_layer_debate_2 = Bidirectional(
            LSTM(lstm_size, return_sequences=True), name='BiDiLSTM Context')(embedded_layer_debate_input2)

    # max-pooling
    max_pool_lambda_layer = Lambda(
        lambda x: keras.backend.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    max_pool_lambda_layer.supports_masking = True
    # two attention vectors

    #    all_input_layers = [bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_debate]
    #    utilized_input_layers = []
    #    for i in range(rich_context):
    #        utilized_input_layers.append(all_input_layers[i])
    #
    #    if rich_embedding:
    #        all_input_layers2 = [bidi_lstm_layer_reason_2, bidi_lstm_layer_claim_2, bidi_lstm_layer_debate_2]
    #        utilized_input_layers2 = []
    #        for i in range(rich_context):
    #            utilized_input_layers2.append(all_input_layers2[i])
    #
    #    layers_w0 = [bidi_lstm_layer_warrant1] + utilized_input_layers
    #    layers_w1 = [bidi_lstm_layer_warrant0] + utilized_input_layers
    #    if rich_embedding:
    #        layers_w0.extend([bidi_lstm_layer_warrant1_2] + utilized_input_layers2)
    #        layers_w1.extend([bidi_lstm_layer_warrant0_2] + utilized_input_layers2)
    #
    #    print('number of input layers w0:', len(layers_w0))
    #    print('number of input layers w1:', len(layers_w1))
    #
    #    if len(layers_w0) > 1:
    #        layers_w0_merge = merge(layers_w0, mode='concat')
    #        layers_w1_merge = merge(layers_w1, mode='concat')
    #    else:
    #        layers_w0_merge = layers_w0
    #        layers_w1_merge = layers_w1
    #
    #    attention_vector_for_w0 = max_pool_lambda_layer(layers_w0_merge)
    #    attention_vector_for_w1 = max_pool_lambda_layer(layers_w1_merge)

    if rich_context:
        attention_vector_for_w0 = max_pool_lambda_layer(
            concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_warrant1, bidi_lstm_layer_debate]))
        attention_vector_for_w1 = max_pool_lambda_layer(
            concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_warrant0, bidi_lstm_layer_debate]))
    else:
        attention_vector_for_w0 = max_pool_lambda_layer(
            concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_warrant1]))
        attention_vector_for_w1 = max_pool_lambda_layer(
            concatenate([bidi_lstm_layer_reason, bidi_lstm_layer_claim, bidi_lstm_layer_warrant0]))

    attention_warrant0 = LSTM(lstm_size)(bidi_lstm_layer_warrant0)
    attention_warrant1 = LSTM(lstm_size)(bidi_lstm_layer_warrant1)

    # concatenate them
    dropout_layer = Dropout(dropout)(concatenate([attention_warrant0, attention_warrant1]))

    # and add one extra layer with ReLU
    # dense1 = Dense(int(lstm_size / 2), activation=activation1)(dropout_layer)
    dense1 = Dense(int(lstm_size / 1), activation=activation1)(dropout_layer)
    # dense2 = Dense(int(lstm_size / 2), activation=activation1)(dense1)
    # dense3 = Dense(int(lstm_size / 4), activation=activation1)(dense2)
    pre_out = dense1
    output_layer = Dense(1, activation=activation2)(pre_out)

    model = Model(inputs=[sequence_layer_warrant0_input, sequence_layer_warrant1_input, sequence_layer_reason_input,
                          sequence_layer_claim_input, sequence_layer_debate_input], outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='tmp/model-lstm.png')

    return model
