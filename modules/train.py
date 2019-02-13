import tensorflow as tf
import random
import numpy as np
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

'''
モジュール
module
'''
def get_key_from_value(d, value):
    keys = [k for k,v in d.items() if v==value]
    if keys:
        return keys[0]
    return '<UNK>' 

def train(global_obj, s2s_g, train_batches, test_batches, vocab_dict, save_path):

    STEPS = global_obj.steps
    MAX_INPUT_SEQUENCE_LENGTH = global_obj.input_length
    MAX_OUTPUT_SEQUENCE_LENGTH = global_obj.output_length
    BATCH_SIZE = global_obj.batch_size
    TEST_SIZE = global_obj.test_size
    LEARNING_RATE = global_obj.learning_rate
    DECAY_RATE = global_obj.decay_rate
    
    # グラフ（Tensorboard）の定義
    graph = tf.Graph()
    with graph.as_default():
        encoder_inputs, decoder_inputs, replies, masks, feed_previous, learning_rate, loss, predictions, optimizer, merged = s2s_g.construct_s2s_graph()

        # セーバー
        if save_path!=None:
            saver = tf.train.Saver(max_to_keep=100)

    # Run session
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        current_learning_rate = LEARNING_RATE 
        if save_path!=None:
            train_path = './'+save_path+'/tensorboard/train'
            os.makedirs(train_path)
            test_path = './'+save_path+'/tensorboard/test'
            os.makedirs(test_path)
            train_writer = tf.summary.FileWriter(train_path, graph)
            test_writer = tf.summary.FileWriter(test_path, graph)

        with open(save_path+"/RESULT.txt", mode='w') as f:
            f.write("steps\ttrain_loss\ttest_loss\n")

        for step in range(STEPS+1):
            current_train_sequences, current_train_tokens, current_train_encoder_inputs, current_train_decoder_inputs, \
                                     current_train_replies, current_train_masks = train_batches.next()

            feed_dict = dict()
            feed_dict = {encoder_inputs[i]: current_train_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
            feed_dict.update({decoder_inputs[i]: current_train_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
            feed_dict.update({replies[i]: current_train_replies[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
            feed_dict.update({masks[i]: current_train_masks[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
            feed_dict.update({feed_previous: False})

            # 学習率を小さくする
            if step != 0 and step % (STEPS/10) == 0:
                current_learning_rate *= DECAY_RATE

            feed_dict.update({learning_rate: current_learning_rate})

            _, current_train_loss, current_train_predictions, train_summary = sess.run([optimizer, loss, predictions, merged], feed_dict=feed_dict)

            if save_path!=None:
                train_writer.add_summary(train_summary, step)
                train_writer.flush()

            if step % (STEPS/100) == 0:

                # softmax → ID → 単語 
                rand = random.randint(0,BATCH_SIZE-1)
                train_output = []
                for i in range(MAX_OUTPUT_SEQUENCE_LENGTH):
                    word_id = np.argmax(current_train_predictions[i][rand])
                    train_output.append(get_key_from_value(vocab_dict, word_id))

                print('Step %d:' % step)
                print('Training set:')
                print('  Loss       : ', current_train_loss)
                print('  Input            : ', current_train_sequences[rand])
                print('Input token        : ', current_train_tokens[rand])
                print('  Generated output : ', train_output)

                current_test_sequences, current_test_tokens, current_test_encoder_inputs, current_test_decoder_inputs, \
                                         current_test_replies, current_test_masks = test_batches.next()
                test_feed_dict = dict()
                test_feed_dict = {encoder_inputs[i]: current_test_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
                test_feed_dict.update({decoder_inputs[i]: current_test_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
                test_feed_dict.update({replies[i]: current_test_replies[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
                test_feed_dict.update({masks[i]: current_test_masks[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH)})
                test_feed_dict.update({feed_previous: True})
                test_feed_dict.update({learning_rate: current_learning_rate})

                current_test_loss, current_test_predictions, test_summary = sess.run([loss, predictions, merged], feed_dict=test_feed_dict)

                rand = random.randint(0,TEST_SIZE-1)

                # softmax → ID → 単語
                rand = random.randint(0,BATCH_SIZE-1)
                test_output = []
                for i in range(MAX_OUTPUT_SEQUENCE_LENGTH):
                    word_id = np.argmax(current_test_predictions[i][rand])
                    test_output.append(get_key_from_value(vocab_dict, word_id))

                print('Test set:')
                print('  Loss       : ', current_test_loss)
                print('  Input            : ', current_test_sequences[rand])
                print('Input token        : ', current_test_tokens[rand])
                print('  Generated output : ', test_output)

                test_writer.add_summary(test_summary, step)
                test_writer.flush()

                if save_path!=None:
                    with open(save_path+"/OUTPUT.log", mode='a') as f:
                        f.write('Output Instance in {} steps :\n'.format(step))
                        f.write('Loss       : {}\n'.format(str(current_test_loss)))
                        f.write('Input            : {}\n'.format(current_test_sequences[rand]))
                        f.write('Input token      : {}\n'.format(current_test_tokens[rand]))
                        f.write('Generated output : {}\n'.format("".join(test_output)))
                        f.write('\n')

                    os.makedirs(save_path+'/checkpoint',exist_ok=True)
                    saver.save(sess,save_path+'/checkpoint/{}steps.ckpt'.format(step))
                    print("Model saved in file: %s" % save_path)
                    with open(save_path+"/RESULT.txt", mode='a') as f:
                        f.write("{:05}\t{:10.7f}\t{:10.7f}\n".format(step, current_train_loss, current_test_loss))
