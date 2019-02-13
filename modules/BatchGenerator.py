import numpy as np
import MeCab

# Utils
def getTokenlist(text):
    token_list = []
    m = MeCab.Tagger('-Owakati')
    token_list =m.parse(text).strip('\n').split(" ")
    return token_list

# バッチ作成
class BatchGenerator(object):

    def __init__(self, global_obj, tweets, replies, vocab_dict, global_id = 0):
        self.BATCH_SIZE = global_obj.batch_size
        self.MAX_INPUT_SEQUENCE_LENGTH = global_obj.input_length
        self.INPUT_REVERSE = global_obj.input_reverse
        self.MAX_OUTPUT_SEQUENCE_LENGTH = global_obj.output_length
        self.GO_ID = global_obj.go_id
        self.PAD_ID = global_obj.pad_id
        self.EOS_ID = global_obj.eos_id
        self.UNK_ID = global_obj.unk_id
        self._GLOBAL_ID = global_id
        self._tweets = tweets
        self._replies = replies
        self._vocab_dict = vocab_dict

    def next(self):
        input_sequences = []
        encoder_inputs = []
        decoder_inputs = []
        decoder_replies = []
        decoder_masks = []

        # グローバルIDリセット
        self._GLOBAL_ID = 0 if self._GLOBAL_ID >= len(self._tweets)-self.BATCH_SIZE and len(self._tweets)!=self.BATCH_SIZE else self._GLOBAL_ID
        for i in range(self.BATCH_SIZE):
            # 入力ベクトル作成
            words = getTokenlist(self._tweets[self._GLOBAL_ID])
            input_unreverse = []
            for i, word in enumerate(words):
                if(i<self.MAX_INPUT_SEQUENCE_LENGTH):
                    input_unreverse.append(self._vocab_dict.get(word, self.UNK_ID)) #トークンから単語IDへ
                else:
                    break
            if self.INPUT_REVERSE:
                encoder_input = [self.PAD_ID]*(self.MAX_INPUT_SEQUENCE_LENGTH-len(input_unreverse)) + input_unreverse[::-1]
            else:
                encoder_input = input_unreverse + [self.PAD_ID]*(self.MAX_INPUT_SEQUENCE_LENGTH-len(input_unreverse))

            # 出力ベクトル作成
            words = getTokenlist(self._replies[self._GLOBAL_ID])
            output_unreverse = []
            for i, word in enumerate(words):
                if(i<self.MAX_OUTPUT_SEQUENCE_LENGTH):
                    output_unreverse.append(self._vocab_dict.get(word, self.UNK_ID)) #トークンから単語IDへ
                else:
                    break
            decoder_reply = output_unreverse + [self.EOS_ID] + [self.PAD_ID]*(self.MAX_OUTPUT_SEQUENCE_LENGTH-len(output_unreverse))
            decoder_input = [self.GO_ID] + output_unreverse + [self.PAD_ID]*(self.MAX_OUTPUT_SEQUENCE_LENGTH-len(output_unreverse))
            decoder_mask = [1.0]*len(output_unreverse)+[1.0] + [0.0]*(self.MAX_OUTPUT_SEQUENCE_LENGTH-len(output_unreverse)) #?

            # append
            input_sequences.append(self._tweets[self._GLOBAL_ID])
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            decoder_replies.append(decoder_reply)
            decoder_masks.append(decoder_mask)
            self._GLOBAL_ID = self._GLOBAL_ID+1
        self._GLOBAL_ID = 0 if len(self._tweets)==self.BATCH_SIZE else self._GLOBAL_ID #テスト時GLOBAL_IDリセット

        return input_sequences, np.array(encoder_inputs).T, np.array(decoder_inputs).T, np.array(decoder_replies).T, np.array(decoder_masks).T

'''メモ
TEST_SIZEとBATCH_SIZEは常に同じにする
'''
