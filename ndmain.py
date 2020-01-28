
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

import nuggetdetectionBERT as ND
import param as param
import stcevaluation as STCE

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses

# global pred
# global x
# global bs
# global turns
# global masks
# global devX
# global devND
# global dev_turns
# global dev_masks
# global dev_corpus
# global sess
# global predict_fn

devX = pickle.load(open('PickleBert/devX_bert_512.p', 'rb'))
devND = pickle.load(open('datacache/devND.p', 'rb'))
dev_turns = pickle.load(open('datacache/dev_turns.p', 'rb'))
dev_masks = pickle.load(open('datacache/dev_masks.p', 'rb'))
dev_corpus = pickle.load(open('datacache/dev_corpus.p', 'rb'))

# Params
batch_size = 30
lr = 5e-4
kp = 1
hiddens = 1024
Fsize = [2, 3]
gating = False
bn = True
method = ND.CNNRNN
memory_rnn_type = None
Fnum = [256]
num_layers = 2


tf.reset_default_graph()
x, y, bs, turns, masks, num_sent = ND.init_input(doclen, embsize)
pred = method(x, y, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, masks, memory_rnn_type)
cost = ND.loss_function(pred, y, batch_size, num_sent, masks)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

predict_fn = predictor.from_saved_model('savemodel/savemodelND')


def predict_examples(n):
    # global pred
    # global x
    # global bs
    # global turns
    # global masks
    # global devX
    # global devND
    # global dev_turns
    # global dev_masks
    # global dev_corpus
    # global sess
    # global predict_fn

    devXin = np.expand_dims(devX[n], axis=0)
    devNDin = np.expand_dims(devND[n], axis=0)
    dev_turns_in = np.expand_dims(dev_turns[n], axis=0)
    dev_masks_in = np.expand_dims(dev_masks[n], axis=0)
    dev_selected_corpus = dev_corpus[n]
    chatid, speakers, contents, _, _ = dev_selected_corpus

    pred_dev = predict_fn({
        "input_X": devXin,
        "output_Y": devNDin,
        "batch_size": 1,
        "turns": dev_turns_in,
        "masks": dev_masks_in,
        "num_sent": 1,
    })["pred"]
    pred_dev = pred_dev.tolist()

    dialog = []

    RNSS, JSD = STCE.nugget_evaluation(pred_dev, devND, dev_turns, dev_masks)
    JSD = '{:.3f}'.format(JSD)
    RNSS = '{:.3f}'.format(RNSS)

    # Add data to render context
    for speaker, content, answer, pred in zip(speakers, contents, devND[0], pred_dev[0]):
        pred = ['{:.3f}'.format(x) for x in pred]
        answer = ['{:.3f}'.format(x) for x in answer]

        utterance = {}
        utterance['speaker'] = speaker
        utterance['content'] = content
        utterance['answer'] = answer
        utterance['pred'] = pred
        dialog.append(utterance.copy())

    context = {
        'dialogid': n,
        'dialog': dialog,
        'JSD': JSD,
        'RNSS': RNSS,
    }

    return context


def predict_dialog(texts, devXin, dev_turns_in, dev_masks_in):
    # global pred
    # global x
    # global bs
    # global turns
    # global masks
    # global devND
    # global dev_corpus

    devNDin = np.expand_dims(devND[0], axis=0)

    pred_dev = predict_fn({
        "input_X": devXin,
        "output_Y": devNDin,
        "batch_size": 1,
        "turns": dev_turns_in,
        "masks": dev_masks_in,
        "num_sent": 1,
    })["pred"]
    pred_dev = pred_dev.tolist()

    dialog = []
    nuggettypes = ['CNUG*', 'CNUG', 'CNaN', 'CNUG0', 'HNUG*', 'HNUG', 'HNaN']

    i = 0
    # Add data to render context
    for content, pred in zip(texts, pred_dev[0]):
        top = pred.index(max(pred))
        pred = ['{:.3f}'.format(x) for x in pred]

        utterance = {}
        speaker = 'Customer' if i % 2 == 0 else 'Helpdesk'
        utterance['speaker'] = speaker
        utterance['content'] = content
        utterance['pred'] = pred
        utterance['top'] = nuggettypes[top]
        dialog.append(utterance.copy())
        i += 1

    context = {
        'dialogid': 'Custom',
        'dialog': dialog,
    }

    return context, pred_dev
