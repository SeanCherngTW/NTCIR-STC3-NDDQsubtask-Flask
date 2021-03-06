import logging

import tensorflow as tf

import param as param

doclen = param.doclen
embsize = param.embsize
sentembsize = param.sentembsize
max_sent = param.max_sent
NDclasses = param.NDclasses
DQclasses = param.DQclasses
logger = logging.getLogger('DQ task')
tf.logging.set_verbosity(tf.logging.ERROR)


def weight_variable(shape, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.get_variable(
            shape=shape,
            initializer=tf.contrib.keras.initializers.he_normal(),
            name=name,
        )


def bias_variable(shape, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.Variable(tf.constant(0.1, shape=shape, name=name))


def conv2d(x, W, embsize, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.nn.conv2d(x, W, strides=[1, 1, embsize, 1], padding='SAME', name=name)


def conv1d(x, filter_size, num_filter, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.layers.conv1d(
            inputs=x,
            filters=num_filter,
            kernel_size=filter_size,
            activation=tf.nn.relu,
            padding='SAME',
            name=name,
        )


def maxpool(h, pool_size, strides, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.layers.max_pooling1d(
            inputs=h,
            pool_size=pool_size,
            strides=strides,
            padding='VALID',
            name=name,
        )


def build_RNN(sentCNNs, bs, turns, rnn_hiddens, batch_norm, name, rnn_type, keep_prob, num_layers):
    def _get_cell(rnn_type, rnn_hiddens):
        assert rnn_type in ['Bi-LSTM', 'Bi-GRU']
        if rnn_type == 'Bi-LSTM':
            return tf.contrib.rnn.BasicLSTMCell(rnn_hiddens, forget_bias=1.0)
        else:
            return tf.contrib.rnn.GRUCell(rnn_hiddens)

    fw_cells = []
    bw_cells = []
    with tf.name_scope(name):
        with tf.name_scope(rnn_type):
            for _ in range(num_layers):
                fw_cell = tf.contrib.rnn.DropoutWrapper(
                    _get_cell(rnn_type, rnn_hiddens),
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                )
                bw_cell = tf.contrib.rnn.DropoutWrapper(
                    _get_cell(rnn_type, rnn_hiddens),
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                )

                fw_cells.append(fw_cell)
                bw_cells.append(bw_cell)

        fw_cells = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_cells = tf.contrib.rnn.MultiRNNCell(bw_cells)
        init_state_fw = fw_cells.zero_state(bs, tf.float32)
        init_state_bw = bw_cells.zero_state(bs, tf.float32)

        (output_fw, output_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cells,
            cell_bw=bw_cells,
            inputs=sentCNNs,
            sequence_length=turns,
            initial_state_fw=init_state_fw,
            initial_state_bw=init_state_bw,
            time_major=False,
            scope=name,
        )

        with tf.name_scope('add_Fw_Bw'):
            rnn_output = tf.nn.tanh(tf.add(output_fw, output_bw))
            logger.debug('{} rnn_output {}'.format(name, str(rnn_output.shape)))

    return rnn_output


def memory_enhanced(rnn_output, input_memory, output_memory):
    with tf.name_scope('memory_enhanced'):
        is_first = True
        rnn_output = tf.unstack(rnn_output, axis=1)
        input_memory = tf.unstack(input_memory, axis=1)
        for sent_t in rnn_output:  # for sentence in time t
            attention_at_time_t = []
            for context_i in input_memory:  # for context sentence i
                # sent_t = (?, 1024), context_i = (?, 1024), _attention = (?, )
                _attention = tf.reduce_sum(tf.multiply(sent_t, context_i), axis=1)
                attention_at_time_t.append(_attention)

            # attention_at_time_t = 7 * (?, ) --stack--> (?, 7), attention_weight_at_time_t = (?, 7)
            # attention_weight_at_time_t = tf.nn.tanh(tf.stack(attention_at_time_t, axis=1))
            attention_weight_at_time_t = tf.nn.softmax(tf.stack(attention_at_time_t, axis=1))
            attention_weight_at_time_t = tf.reshape(attention_weight_at_time_t, [-1, max_sent, 1])  # boardcast weight

            # attention_weight_at_time_t = (?, 7), output_memory = (?, 7, 1024), weighted_output_memory = (?, 7, 1024)
            logger.debug('attention_weight_at_time_t {}'.format(str(attention_weight_at_time_t.shape)))
            logger.debug('output_memory {}'.format(str(output_memory.shape)))
            weighted_output_memory = tf.multiply(output_memory, attention_weight_at_time_t)
            logger.debug('weighted_output_memory {}'.format(str(weighted_output_memory.shape)))

            # weighted_output_memory = (?, 7, 1024), weighted_sum = (?, 1024)
            weighted_sum = tf.reduce_sum(weighted_output_memory, axis=1)

            # sent_t_with_memory = (?, 1024)
            sent_t_with_memory = tf.add(weighted_sum, sent_t)
            sent_t_with_memory = tf.expand_dims(sent_t_with_memory, axis=1)

            if is_first:
                sents_with_memory = sent_t_with_memory
                is_first = False
            else:
                sents_with_memory = tf.concat([sents_with_memory, sent_t_with_memory], axis=1)

    return sents_with_memory  # (?, 7, 1024)


def init_input(doclen, embsize):
    # doclen = 150, embsize = 256
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, max_sent, sentembsize], name='input_X')
        y = tf.placeholder(tf.float32, [None, DQclasses], name='output_Y')
        bs = tf.placeholder(tf.int32, [], name='batch_size')
        turns = tf.placeholder(tf.int32, [None, ], name='turns')
        num_dialog = tf.placeholder(tf.int32, [], name='num_dialog')
        nd = tf.placeholder(tf.float32, [None, max_sent, NDclasses], name='nd')
    return x, y, bs, turns, num_dialog, nd


def CNNCNN(x, bs, turns, keep_prob, fc_hiddens, filter_size, num_filters, gating, batch_norm, num_layers, nd, memory_rnn_type=None):
    x_split = tf.unstack(x, axis=1)
    nd_split = tf.unstack(nd, axis=1)

    for i in range(max_sent):
        if i % 2 == 0:  # customer
            speaker = tf.fill((bs, 1, 1), 0.0)
        else:  # helpdesk
            speaker = tf.fill((bs, 1, 1), 1.0)
        x_split_expand = tf.expand_dims(x_split[i], axis=1)
        nd_split_expand = tf.expand_dims(nd_split[i], axis=1)
        concated = tf.concat([x_split_expand, speaker, nd_split_expand], axis=-1, name='speaker_concated')
        if i == 0:
            sentCNNs = concated
        else:
            sentCNNs = tf.concat([sentCNNs, concated], axis=1, name='sentCNN_concate_sent')
    sentCNNs = tf.unstack(sentCNNs, axis=1)  # (?, filter_num) * 7

    _contextCNNs = []
    contextCNNs_reuse = False
    is_first = True
    for i in range(max_sent):
        if i == 0:
            start = tf.fill((bs, sentCNNs[i].shape[-1]), 0.0)
            _contextCNNs.append(tf.concat([start, sentCNNs[i], sentCNNs[i + 1]], axis=-1))
        elif i == max_sent - 1:
            end = tf.fill((bs, sentCNNs[i].shape[-1]), 0.0)
            _contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], end], axis=-1))
        else:
            _contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], sentCNNs[i + 1]], axis=-1))

    # contextCNNs = (?, 3, filter_num) * 7

    for i in range(max_sent):
        logger.debug('_contextCNNs shape from {}'.format(_contextCNNs[i].shape))
        _, filters = _contextCNNs[i].shape
        _contextCNNs[i] = tf.reshape(_contextCNNs[i], [-1, 1, filters])
        logger.debug('_contextCNNs shape to {}'.format(_contextCNNs[i].shape))

    num_filters = [1024] * num_layers

    # Context CNNs
    for i, x_context in enumerate(_contextCNNs):
        if gating:
            for layer, Fnum in enumerate(num_filters):
                contextCNN_convA = conv1d(x_context, filter_size[0], Fnum, 'contextCNN_convA{}'.format(layer), contextCNNs_reuse)
                contextCNN_convB = conv1d(x_context, filter_size[1], Fnum, 'contextCNN_convB{}'.format(layer), contextCNNs_reuse)
                x_context = tf.multiply(contextCNN_convA, tf.nn.sigmoid(contextCNN_convB), name='context_gating{}'.format(layer))
                if batch_norm:
                    x_context = tf.layers.batch_normalization(x_context)

            # (?, 1, 128) -> (?, 1, 64)
            contextCNN_pool = maxpool(x_context, 1, 2, 'contextCNN_pool', contextCNNs_reuse)
            concated = contextCNN_pool

        else:
            contextCNN_convA = x_context
            contextCNN_convB = x_context
            for layer, Fnum in enumerate(num_filters):
                contextCNN_convA = conv1d(contextCNN_convA, filter_size[0],
                                          Fnum, 'contextCNN_convA{}'.format(layer), contextCNNs_reuse)
                contextCNN_convB = conv1d(contextCNN_convB, filter_size[1],
                                          Fnum, 'contextCNN_convB{}'.format(layer), contextCNNs_reuse)
                if batch_norm:
                    contextCNN_convA = tf.layers.batch_normalization(contextCNN_convA)
                    contextCNN_convB = tf.layers.batch_normalization(contextCNN_convB)

            # (?, 1, 128) -> (?, 1, 64)
            contextCNN_poolA = maxpool(contextCNN_convA, 1, 2, 'contextCNN_poolA', contextCNNs_reuse)
            contextCNN_poolB = maxpool(contextCNN_convB, 1, 2, 'contextCNN_poolB', contextCNNs_reuse)
            concated = tf.concat([contextCNN_poolA, contextCNN_poolB], axis=-1)

        if is_first:
            contextCNNs = concated
            contextCNNs_reuse = True
            is_first = False
        else:
            contextCNNs = tf.concat([contextCNNs, concated], axis=1)

    logger.debug('contextCNNs output shape {}'.format(str(contextCNNs.shape)))

    if memory_rnn_type:
        input_memory = build_RNN(contextCNNs, bs, turns, fc_hiddens, batch_norm,
                                 'input_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        output_memory = build_RNN(contextCNNs, bs, turns, fc_hiddens, batch_norm,
                                  'output_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        contextCNNs = memory_enhanced(contextCNNs, input_memory, output_memory)

    logger.debug('contextCNNs output shape {}'.format(str(contextCNNs.shape)))
    _, num_sent, num_features = contextCNNs.shape
    contextCNNs = tf.reshape(contextCNNs, [-1, num_sent * num_features])
    logger.debug('FC input shape {}'.format(str(contextCNNs.shape)))

    contextCNNs = tf.nn.dropout(contextCNNs, keep_prob)

    # Fully Connected Layer
    fc1_W = weight_variable([contextCNNs.shape[-1], fc_hiddens], name='fc1_W')
    fc1_b = bias_variable([fc_hiddens, ], name='fc1_b')
    fc1_out = tf.nn.relu(tf.matmul(contextCNNs, fc1_W) + fc1_b)

    if batch_norm:
        fc1_out = tf.layers.batch_normalization(fc1_out)

    fc2_W = weight_variable([fc_hiddens, DQclasses], name='fc2_W')
    fc2_b = bias_variable([DQclasses, ], name='fc2_b')
    fc2_out = tf.matmul(fc1_out, fc2_W) + fc2_b

    # y_pre = fc2_out
    y_pre = tf.nn.softmax(fc2_out)

    return y_pre
