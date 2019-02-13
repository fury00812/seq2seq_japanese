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
