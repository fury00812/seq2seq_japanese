import sys
import os
import numpy as np
import random
import collections
import pickle
from six.moves.urllib.request import urlretrieve
from pathlib import Path 
import datetime
import MeCab
import argparse
import configparser
import shutil
from distutils.util import strtobool
from modules import BatchGenerator as batch_class
from modules import GraphConstructor as graph_class
from modules import train

sys.path.append('../')
from module.prepare_trainingData import prepare_trainingData as pt

'''
グローバル変数
global variables 
'''
VOCAB_NUM = 50000
VOCAB_TYPE = 'Many'
LABEL_NUM = 600 
BATCH_SIZE = 500
TEST_SIZE = 500
MAX_INPUT_SEQUENCE_LENGTH = 7 
MAX_OUTPUT_SEQUENCE_LENGTH = 7 
GO_ID = 0 
PAD_ID = 1 
EOS_ID = 2 
UNK_ID = 3
INPUT_REVERSE = False
LSTM_SIZE = 256 
ATTENTION = True 
EMBEDDING_SIZE = 128
STEPS = 10000 #BATCH_SIZE500のとき、約60step=1エポックである
NAZOPARAM = 10
LEARNING_RATE = 0.1
DECAY_RATE = 0.9

'''
モジュール
modules
'''
def update_params(labeldict):
    global LABEL_NUM, GO_ID, PAD_ID, EOS_ID
    LABEL_NUM = len(labeldict)+NAZOPARAM
    GO_ID = LABEL_NUM + 1
    PAD_ID = LABEL_NUM + 2
    EOS_ID = LABEL_NUM + 3

def get_save_path(output_dir):
    dt_now = '{0:%Y%m%d%H%M}'.format(datetime.datetime.now())    
    save_path = 'models/{}-{}'.format(output_dir,dt_now)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def shuffleData(raw_tweets, raw_replies):
    zipped = list(zip(raw_tweets,raw_replies))
    np.random.shuffle(zipped)
    s,l = zip(*zipped)
    raw_tweets = list(s)
    raw_replies = list(l)
    return raw_tweets, raw_replies 

def makeTestData(raw_tweets, raw_replies):
    test_tweets = raw_tweets[:TEST_SIZE]
    test_replies = raw_replies[:TEST_SIZE]
    train_tweets = raw_tweets[TEST_SIZE:]
    train_replies = raw_replies[TEST_SIZE:]
    return test_tweets, test_replies, train_tweets, train_replies

def get_vocab_dict(all_vocab_dict, vocab_num, vocab_type, aizuchi_l):
    vocab_dict = {}
    vocab_dict['<GO>'] = GO_ID 
    vocab_dict['<PAD>'] = PAD_ID 
    vocab_dict['<EOS>'] = EOS_ID 
    print('ALL_VOCABURALY',len(all_vocab_dict))

    # 相槌を辞書に追加する
    for aizuchi in aizuchi_l:
        vocab_dict[aizuchi] = len(vocab_dict)

    if vocab_type=='Many':
        for k,v in all_vocab_dict.items():
            vocab_dict[k] = len(vocab_dict) 
            if len(vocab_dict) >= vocab_num+4:
                break
    elif vocab_type=='Few': 
        for k,v in sorted(all_vocab_dict.items(), key=lambda x: x[1]):
            vocab_dict[k] = len(vocab_dict)
            if len(vocab_dict) >= vocab_num+4:
                break
    elif vocab_type=='Middle' and len(all_vocab_dict) >= vocab_num:
        offset = len(all_vocab_dict)/2 - vocab_num/2
        for i, (k,v) in enumerate(all_vocab_dict.items()):
            if i>=offset:
                vocab_dict[k] = len(vocab_dict) 
            if len(vocab_dict) >= vocab_num+4:
                break
    return vocab_dict

'''
グローバル変数の共有
sharing global variables class 
'''
class GlobalInfo:
    def __init__(self, param_path):
        self.inifile = configparser.ConfigParser()
        if param_path!=None:
            self.inifile.read(param_path)
        self.vocab_num = int(self.inifile.get('params', 'VOCAB_NUM')) if param_path!=None else VOCAB_NUM
        self.vocab_type = str(self.inifile.get('params', 'VOCAB_TYPE')) if param_path!=None else VOCAB_TYPE
        self.batch_size = int(self.inifile.get('params','BATCH_SIZE')) if param_path!=None else BATCH_SIZE
        self.test_size = int(self.inifile.get('params','TEST_SIZE')) if param_path!=None else TEST_SIZE
        self.input_length = int(self.inifile.get('params','MAX_INPUT_SEQUENCE_LENGTH')) if param_path!=None else MAX_INPUT_SEQUENCE_LENGTH
        self.output_length = int(self.inifile.get('params','MAX_OUTPUT_SEQUENCE_LENGTH')) if param_path!=None else MAX_OUTPUT_SEQUENCE_LENGTH
        self.go_id = GO_ID 
        self.pad_id = PAD_ID 
        self.eos_id = EOS_ID 
        self.unk_id = UNK_ID
        self.input_reverse = strtobool(self.inifile.get('params','INPUT_REVERSE')) if param_path!=None else INPUT_REVERSE
        self.lstm_size = int(self.inifile.get('params','LSTM_SIZE')) if param_path!=None else LSTM_SIZE
        self.attention = strtobool(self.inifile.get('params','ATTENTION')) if param_path!=None else ATTENTION
        self.embedding_size = int(self.inifile.get('params','LSTM_SIZE')) if param_path!=None else EMBEDDING_SIZE
        self.steps = int(self.inifile.get('params','STEPS')) if param_path!=None else STEPS
        self.learning_rate = float(self.inifile.get('params','LEARNING_RATE')) if param_path!=None else LEARNING_RATE
        self.decay_rate = float(self.inifile.get('params','DECAY_RATE')) if param_path!=None else DECAY_RATE
        self.nazo = int(self.inifile.get('params','NAZOPARAM')) if param_path!=None else NAZOPARAM

    def update_vocab_num(self,vocab_dict):
        self.vocab_num = len(vocab_dict)

'''
メインプロセス
main process
'''
def main(dir_path, output_dir, param_path):

    # グローバル変数オブジェクト
    global_obj = GlobalInfo(param_path)

    # データの読み込み
    raw_tweets, raw_replies = shuffleData(pt.pickle_load(dir_path+'raw_tweets.pickle'), pt.pickle_load(dir_path+'raw_replies.pickle'))

    print('NUMBER OF SENTENCE : ',len(raw_tweets))

    # 相槌リストの作成
    aizuchi_l = pt.pickle_load(dir_path+'aizuchi_list.pickle')

    # 学習,テストデータ・語彙辞書の作成
    test_tweets, test_replies, train_tweets, train_replies = makeTestData(raw_tweets, raw_replies)    
    vocab_dict = get_vocab_dict(pt.pickle_load(dir_path+'vocab_dict.pickle'), global_obj.vocab_num, global_obj.vocab_type, aizuchi_l)
    global_obj.update_vocab_num(vocab_dict)

    print('VOCABULARY SIZE : ', global_obj.vocab_num)
 
    # バッチオブジェクトの作成
    train_batches = batch_class.BatchGenerator(global_obj, train_tweets, train_replies, vocab_dict)
    test_batches = batch_class.BatchGenerator(global_obj, test_tweets, test_replies, vocab_dict)

    # モデルの構築
    s2s_g = graph_class.GraphConstructor(global_obj)

    # モデル学習
    save_path = None # モデルを保存する場合のみ 
    if output_dir!=None:
        save_path = get_save_path(output_dir)
        shutil.copyfile(param_path, save_path+'/PARAMETER.ini')  
        with open(save_path+'/PARAMETER.ini', mode='a') as f:
            f.write('[result]\nNUMBER_OF_SENTENCE={}\nVOCABULARY_SIZE={}'.format(str(len(raw_tweets)),str(global_obj.vocab_num)))
    train.train(global_obj, s2s_g, train_batches, test_batches, vocab_dict, save_path)

    # モデル学習・保存
    #dirpath = get_dirpath()
    #train.train(train_batches, test_batches, label_dict, dirpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dir_path', help='train data directory path', type=str)
    parser.add_argument('--o',dest='output_dir', help='output directory name', type=str)
    parser.add_argument('--p', dest='param_path', help='option : assign parameter file', type=str)
    args = parser.parse_args()

    main(args.dir_path,args.output_dir,args.param_path)
