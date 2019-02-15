# About
日本語のseq2seq発話モデル作成プログラム

## main.py
```
python main.py -d '../data/trainData/top1-100k_center/' --o 'top1-100k_center' --p 'params.ini'
```

- コマンドライン引数  
`-d` : 学習データのpickleのパス(prepare_trainingDataで作成)  
`--o` : モデルを保存するディレクトリ名（オプション）. 未使用でモデルを保存しない  
`--p` : パラメータ設定ファイルのパス（オプション）. 未使用でデフォルトパラメータを使用. oを指定する場合必ず指定 

## params.ini
パラメータを記述する設定ファイル

# Branch "word2vec"
学習済みのword2vecモデルをロードして単語embeddingに用いる.  
word2vecモデルは, prepare_trainingData(Branch "regression")で作成する,   
語彙辞書に対応した[VOCAB_NUM×ベクトルサイズ]のndarrayをpickle化したものを読み込む.   

## 使用法
`tensorflow-modules/`下の2つのファイルを以下のように移動させる

`seq2seq.py`
```
/home/user/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
```

`core_rnn_cell.py`
```
/home/rfukuda/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py
```
