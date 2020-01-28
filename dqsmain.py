
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

import dialogquality_ndfeatureBERT as DQNDF
import param as param
import stcevaluation as STCE

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses

devX = pickle.load(open('PickleBert/devX_bert_512.p', 'rb'))
devND = pickle.load(open('datacache/devND.p', 'rb'))
devDQS = pickle.load(open('datacache/devDQS.p', 'rb'))
dev_turns = pickle.load(open('datacache/dev_turns.p', 'rb'))

lr = 5e-4
kp = 1
hiddens = 1024
Fsize = [2, 2]
gating = True
bn = True
memory_rnn_type = 'Bi-GRU'
Fnum = [512, 1024]
num_layers = 1
context = {}

tf.reset_default_graph()
x, y, bs, turns, num_dialog, nd = DQNDF.init_input(doclen, embsize)
pred = DQNDF.CNNCNN(x, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, nd, memory_rnn_type)
cost = tf.divide(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))), tf.cast(num_dialog, tf.float32))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

predict_fn = predictor.from_saved_model('savemodel/savemodelDQS')


def predict_examples(n):

    scoretype = 'DQS'

    devXin = np.expand_dims(devX[n], axis=0)
    devNDin = np.expand_dims(devND[n], axis=0)
    dev_turns_in = np.expand_dims(dev_turns[n], axis=0)
    devYin = np.expand_dims(devDQS[n], axis=0)

    pred_dev = predict_fn({
        "input_X": devXin,
        "output_Y": devYin,
        "batch_size": 1,
        "turns": dev_turns_in,
        "num_dialog": 1,
        "nd": devNDin,
    })["pred"]
    pred_dev = pred_dev.tolist()

    DQscale = [-2, -1, 0, 1, 2]
    predAVG = 0
    ansAVG = 0
    for s, p, a in zip(DQscale, pred_dev[0], devYin[0]):
        predAVG += s * p
        ansAVG += s * a

    context['{}_pred_AVG'.format(scoretype)] = '{:.3f}'.format(predAVG)
    context['{}_ans_AVG'.format(scoretype)] = '{:.3f}'.format(ansAVG)

    pred = ['{:.3f}'.format(x) for x in pred_dev[0]]
    answer = ['{:.3f}'.format(x) for x in devYin[0]]

    NMD, RSNOD = STCE.quality_evaluation(pred_dev, devYin)
    NMD = '{:.3f}'.format(NMD)
    RSNOD = '{:.3f}'.format(RSNOD)

    context['{}_pred'.format(scoretype)] = pred
    context['{}_answer'.format(scoretype)] = answer
    context['{}_NMD'.format(scoretype)] = NMD
    context['{}_RSNOD'.format(scoretype)] = RSNOD

    return context


def predict_dialog(devXin, pred_devND, dev_turns_in):
    scoretype = 'DQS'
    devYin = np.expand_dims(devDQS[0], axis=0)

    pred_dev = predict_fn({
        "input_X": devXin,
        "output_Y": devYin,
        "batch_size": 1,
        "turns": dev_turns_in,
        "num_dialog": 1,
        "nd": pred_devND,
    })["pred"]
    pred_dev = pred_dev.tolist()

    context = {}
    DQscale = [-2, -1, 0, 1, 2]
    predAVG = 0
    for s, p in zip(DQscale, pred_dev[0]):
        predAVG += s * p

    context['{}_AVG'.format(scoretype)] = '{:.3f}'.format(predAVG)
    pred = ['{:.3f}'.format(x) for x in pred_dev[0]]

    context['{}_pred'.format(scoretype)] = pred

    return context
