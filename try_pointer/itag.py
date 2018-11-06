# -*- coding=utf8 -*-
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Embedding, TimeDistributed, Dense, Dropout, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from attention import TimeAttention, Masked, Attention
import shared_dataset
import utils
import numpy as np
from keras import backend as K
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ALL_WORDS = 20705
WORD_VOCAB = 20000
LABEL_VOCAB = 2500
DE_TOKENS = LABEL_VOCAB - 2  # remove pad and start tokens from file_vocab
MAX_WORDS = 100
MAX_LABELS = 5

INDEX_FROM = 3
END_TOKEN = 20704
START_TOKEN = 20705
LABEL_FROM = 18215

GRU_SIZE = 256
ATTENTION_SIZE = 256
EMBEDDING_DIM = 100

KEEP_PROB = 0.1
NUM_EPOCHS = 60
BATCH_SIZE = 100

TOPIC_NUM = 100
BEAM_SIZE = 3
MAX_LENGTH = 5


def mean_negative_log_probs(y_true, y_pred):
    '''

    :param y_true: 
    :param y_pred: 
    :return: 
    '''
    log_probs = -K.log(y_pred)
    log_probs *= y_true
    return K.sum(log_probs) / K.sum(y_true)


def compute_precision(y_true, y_pred):
    '''
    Customized precision metric for Keras model log
    Input: y_true, y_pred
    Output: precision score
    '''

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def compute_recall(y_true, y_pred):
    '''
    Customized recall metric for Keras model log
    Input: y_true, y_pred
    Output: recall score
    '''

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# model
encoder_input = Input(shape=(MAX_WORDS,))
decoder_input = Input(shape=(MAX_LABELS,))

shared_embedded = Embedding(ALL_WORDS, EMBEDDING_DIM, mask_zero=True)
encoder_embedded = shared_embedded(encoder_input)
encoder_outputs, state1 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')(encoder_embedded)
encoder_outputs = Dropout(KEEP_PROB)(encoder_outputs)
encoder_outputs, state2 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')(encoder_outputs)
encoder_outputs = Dropout(KEEP_PROB)(encoder_outputs)
encoder_outputs, state3 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')(encoder_outputs)

encoder_outputs = Masked()(encoder_outputs)

decoder_outputs = shared_embedded(decoder_input)
decoder_gru1 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')
decoder_outputs, n_state = decoder_gru1(decoder_outputs, initial_state=state1)
decoder_outputs = Dropout(KEEP_PROB)(decoder_outputs)
decoder_gru2 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')
decoder_outputs, n_state = decoder_gru2(decoder_outputs, initial_state=state2)
decoder_outputs = Dropout(KEEP_PROB)(decoder_outputs)
decoder_gru3 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros')
decoder_outputs, n_state = decoder_gru3(decoder_outputs, initial_state=state3)
time_attention = TimeAttention(units=ATTENTION_SIZE, return_alphas=True)
decoder_outputs, decoder_alphas, decoder_pgen = time_attention([encoder_outputs, decoder_outputs])

decoder_outputs = Dropout(KEEP_PROB)(decoder_outputs)
decoder_dense = Dense(DE_TOKENS, activation='softmax')
weight_dense = Dense(MAX_WORDS, activation='softmax')
pgen_dense = Dense(1, activation='softmax')
y_ = decoder_dense(decoder_outputs)
w_ = weight_dense(decoder_alphas)
p_ = pgen_dense(decoder_pgen)

model = Model(inputs=[encoder_input, decoder_input], outputs=[y_, w_])
# compile model
model.compile(optimizer='adam', loss=mean_negative_log_probs, metrics=[compute_precision, compute_recall])

# load data
(en_train, ms_train, de_train, y_train), (en_test, ms_test, de_test, y_test) = \
    shared_dataset.load_data(path='../askubuntu/au_259740.npz', num_words=WORD_VOCAB, num_sfs=LABEL_VOCAB, start_sf=START_TOKEN, end_sf=END_TOKEN, sf_len=MAX_LABELS)
en_train = pad_sequences(en_train, padding='post', truncating='post', maxlen=MAX_WORDS)
de_train = pad_sequences(de_train, padding='post', truncating='post', maxlen=MAX_LABELS)
y_train = pad_sequences(y_train, padding='post', truncating='post', maxlen=MAX_LABELS)
w_train = np.array([utils.weight_one_hot(en_train[i], y_train[i]) for i in range(len(y_train))])
y_train = np.array([utils.encode_one_hot(y - LABEL_FROM + 2, DE_TOKENS) for y in y_train])

en_test = pad_sequences(en_test, padding='post', truncating='post', maxlen=MAX_WORDS)
de_test = pad_sequences(de_test, padding='post', truncating='post', maxlen=MAX_LABELS)
y_test = pad_sequences(y_test, padding='post', truncating='post', maxlen=MAX_LABELS)
yo_test = np.array([utils.encode_one_hot(y - LABEL_FROM + 2, DE_TOKENS) for y in y_test])

# train model
es = EarlyStopping(monitor='val_loss', patience=2)
cp = ModelCheckpoint(filepath='itag.h5', monitor='val_loss', save_best_only=True)
model.fit([en_train, de_train],
          [y_train, w_train], validation_split=0.1, epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[es, cp], verbose=2)
model.load_weights('itag.h5')

# predict
encoder_model = Model([encoder_input], [encoder_outputs, state1, state2, state3])

en_state1 = Input(shape=(GRU_SIZE,))
en_state2 = Input(shape=(GRU_SIZE,))
en_state3 = Input(shape=(GRU_SIZE,))
de_context = Input(shape=(MAX_WORDS, GRU_SIZE,))
current_token = Input(shape=(1,))

decoder_out = shared_embedded(current_token)
decoder_out, de_state1 = decoder_gru1(decoder_out, initial_state=en_state1)
decoder_out, de_state2 = decoder_gru2(decoder_out, initial_state=en_state2)
decoder_out, de_state3 = decoder_gru3(decoder_out, initial_state=en_state3)
decoder_out, decoder_al, decoder_pg = time_attention([de_context, decoder_out])
decoder_out = TimeDistributed(decoder_dense)(decoder_out)
decoder_al = weight_dense(decoder_al)
decoder_pg = pgen_dense(decoder_pg)
decoder_model = Model([current_token, de_context, en_state1, en_state2, en_state3],
                      [decoder_out, decoder_al, decoder_pg, de_state1, de_state2, de_state3])


def predict_next_token(en, current, full_context, en_st1, en_st2, en_st3, cur_depth, joint_prs, res, tags):
    cur_depth += 1
    prs, weights, pgen, en_st1, en_st2, en_st3 = decoder_model.predict([current, full_context, en_st1, en_st2, en_st3])

    prs = prs[0, 0, :]
    new_prs = []
    for pr in prs:
        new_prs.append(pr * pgen)
    for i in range(len(en)):
        if (en[i] - LABEL_FROM) > 0 and (en[i] - LABEL_FROM) < DE_TOKENS:
            new_prs[en[i] - LABEL_FROM] = new_prs[en[i] - LABEL_FROM] + weights[0][0][i] * (1 - pgen)
    prs = new_prs
    prs = [(i + 2, v) for i, v in zip(xrange(len(prs)), prs)]
    prs = sorted(prs, lambda x, y: cmp(x[1], y[1]) / cur_depth, reverse=True)

    for p in prs[:BEAM_SIZE]:
        if cur_depth == MAX_LENGTH:
            if p[0] not in tags:
                res.append(((joint_prs + p[1]) / cur_depth, tags[:] + [p[0]]))
        else:
            if p[0] not in tags:
                token = np.zeros((1, 1))
                token[0, 0] = p[0] + LABEL_FROM - 2
                predict_next_token(en, token, full_context, en_st1, en_st2, en_st3,
                                   cur_depth, joint_prs + np.log(p[1]), res, tags[:] + [p[0]])
        if cur_depth == MAX_LENGTH:
            break


# prediction
full_hit_count = 0
recall = 0
precise = 0
count = 0

for (en, y) in zip(en_test, y_test):
    count += 1
    context, en_state1, en_state2, en_state3 = encoder_model.predict(np.array([en]))

    cur_token = np.zeros((1, 1))
    cur_token[0, 0] = START_TOKEN
    results = []
    predict_next_token(en, cur_token, context, en_state1, en_state2, en_state3, 0, 0.0, results, [])
    results = sorted(results, lambda x, y: cmp(x[0], y[0]), reverse=True)
    if len(results) == 0:
        continue
    decoder_seq = results[0][1]
    decoder_seq = [w + LABEL_FROM - 2 for w in decoder_seq]

    y = list(y)
    if count % 1000 == 0:
        print(count)
        print(decoder_seq, y)
    tmp_precision = 0
    tmp_recall = 0

    intersection = list(set(y).intersection(set(decoder_seq)))
    if END_TOKEN in intersection:
        intersection.remove(END_TOKEN)
    y_set = set(y)
    if 0 in y_set:
        y_set.remove(0)
    if END_TOKEN in y_set:
        y_set.remove(END_TOKEN)
    if len(intersection) > 0:
        tmp_recall = len(intersection) * 1.0 / len(y_set)
        recall += tmp_recall

    decoder_seq_set = set(decoder_seq)
    if END_TOKEN in decoder_seq_set:
        decoder_seq_set.remove(END_TOKEN)
    if len(intersection) > 0:
        tmp_precision = len(intersection) * 1.0 / len(decoder_seq_set)
        # precise += len(intersection) * 1.0 / 5
        precise += tmp_precision

    tmp_f1 = 0
    if tmp_recall != 0 or tmp_precision != 0:
        tmp_f1 = 2 * tmp_precision * tmp_recall / (tmp_precision + tmp_recall)


    isHit = True
    for d, yl in zip(decoder_seq, y):
        if d != yl:
            isHit = False
            break
    if isHit:
        full_hit_count += 1

full_hit_count /= len(en_test) * 1.0
precise /= len(en_test) * 1.0
recall /= len(en_test) * 1.0
f1score = 2 * precise * recall / (precise + recall)
print("full hit: %f, precision@%d: %f, recall@%d: %f, f1@%d: %f" % (full_hit_count, MAX_LENGTH, precise, MAX_LENGTH, recall, MAX_LENGTH, f1score))
