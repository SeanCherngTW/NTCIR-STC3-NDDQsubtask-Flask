
import flask
import numpy as np

import dqamain
import dqemain
import dqsmain
import ndmain

app = flask.Flask(__name__)


@app.route('/index', methods=['GET'])
def index():
    return flask.render_template('index.html')


@app.route('/show_examples', methods=['POST'])
def show_examples():
    if flask.request.method == 'POST':
        n = flask.request.values['CIdx']
    ndcontext = ndmain.predict_examples(int(n))
    dqacontext = dqamain.predict_examples(int(n))
    dqecontext = dqemain.predict_examples(int(n))
    dqscontext = dqsmain.predict_examples(int(n))
    context = ndcontext.copy()
    context.update(dqacontext)
    context.update(dqecontext)
    context.update(dqscontext)
    return flask.render_template("result.html", **context)


@app.route('/create_dialog', methods=['POST'])
def create_dialog():

    texts = []
    for i in range(1, 8):
        name = 'utt{}'.format(i)
        if flask.request.values[name]:
            texts.append(flask.request.values[name])
        else:
            break

    if type(texts) != list:
        assert False, 'Input should be a list'

    from bert_serving.client import BertClient
    bc = BertClient(ip='140.115.54.42')

    turns = []
    bertX = []
    dialogbertX = []

    for text in texts:
        text = '.' if text == '' else text
        vec = np.reshape(bc.encode([text]), 1024)
        dialogbertX.append(vec)

    turns.append(len(dialogbertX))

    # Pending with zero for dialogs with turns < 7
    while len(dialogbertX) < 7:
        dialogbertX.append(np.zeros([1024, ]))

    bertX.append(np.asarray(dialogbertX))
    masks = turn2mask(turns)
    ndcontext, pred_devND = ndmain.predict_dialog(texts, np.asarray(bertX), turns, masks)
    dqacontext = dqamain.predict_dialog(np.asarray(bertX), pred_devND, turns)
    dqecontext = dqemain.predict_dialog(np.asarray(bertX), pred_devND, turns)
    dqscontext = dqsmain.predict_dialog(np.asarray(bertX), pred_devND, turns)

    context = ndcontext.copy()
    context.update(dqacontext)
    context.update(dqecontext)
    context.update(dqscontext)

    return flask.render_template("resultcustom.html", **context)


def turn2mask(turns):
    # {'CNUG*': 0, 'CNUG': 1, 'CNaN': 2, 'CNUG0': 3, 'HNUG*': 4, 'HNUG': 5, 'HNaN': 6}
    max_sent = 7
    all_dialog_masks = []
    for turn in turns:
        dialog_mask = []
        for i in range(max_sent):
            if i < turn:
                if i % 2 == 0:  # customer
                    dialog_mask.append(np.concatenate((np.ones(4), np.zeros(3))))
                else:  # helpdesk
                    dialog_mask.append(np.concatenate((np.zeros(4), np.ones(3))))
            else:
                dialog_mask.append(np.zeros(max_sent))

        dialog_mask = np.asarray(dialog_mask)
        all_dialog_masks.append(np.asarray(dialog_mask.copy()))
    return all_dialog_masks


if __name__ == '__main__':
    app.debug = True
    app.run()
