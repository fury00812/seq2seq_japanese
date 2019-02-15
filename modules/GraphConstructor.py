import tensorflow as tf

class GraphConstructor(object):
    def __init__(self, global_obj, w2v_array):
        self.VOCAB_NUM = global_obj.vocab_num
        self.MAX_INPUT_SEQUENCE_LENGTH = global_obj.input_length
        self.MAX_OUTPUT_SEQUENCE_LENGTH = global_obj.output_length
        self.LSTM_SIZE = global_obj.lstm_size
        self.ATTENTION = global_obj.attention
        self.EMBEDDING_SIZE = global_obj.embedding_size

        self.encoder_inputs = list()
        self.decoder_inputs = list()
        self.replies = list()
        self.masks = list()
        self.feed_previous = None
        self.learning_rate = None
        self.loss = None
        self.predictions = None
        self.merged = None
        self.w2v_array = w2v_array

    def construct_s2s_graph(self):
        for _ in range(self.MAX_INPUT_SEQUENCE_LENGTH):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[300,1]))
        for _ in range(self.MAX_OUTPUT_SEQUENCE_LENGTH):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
            self.replies.append(tf.placeholder(tf.int32, shape=(None, )))
            self.masks.append(tf.placeholder(tf.float32, shape=(None, )))
    
            self.feed_previous = tf.placeholder(tf.bool) #self._decoder_inputsを使用するか,学習中はFalse
            self.learning_rate = tf.placeholder(tf.float32) #学習率
    
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.LSTM_SIZE)
        with tf.variable_scope("seq2seq"):
            if self.ATTENTION:
                outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(self.encoder_inputs,
                                                                                        self.decoder_inputs,
                                                                                        cell,
                                                                                        self.VOCAB_NUM,
                                                                                        self.VOCAB_NUM,
                                                                                        self.EMBEDDING_SIZE,
                                                                                        self.w2v_array,
                                                                                        feed_previous=self.feed_previous,
                                                                                       )
            else:
                outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self._encoder_inputs,
                                                                                        self._decoder_inputs,
                                                                                        cell,
                                                                                        self.VOCAB_NUM,
                                                                                        self.VOCAB_NUM,
                                                                                        self.EMBEDDING_SIZE,
                                                                                        feed_previous=self.feed_previous
                                                                                       )
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, self.replies, self.masks)
        self.predictions = tf.stack([tf.nn.softmax(output) for output in outputs])
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
    
        tf.summary.scalar('learning rate', self.learning_rate)
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
    
        return self.encoder_inputs, self.decoder_inputs, self.replies, self.masks, self.feed_previous, \
               self.learning_rate, self.loss, self.predictions, self.optimizer, self.merged
             
