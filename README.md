# othello_neural

# やること
## neural networkの構築

5 layer くらいで最初やってみる

それぞれのニューロンの数

64 100 150 100 30 3

input X はボードの状態
教師の方に関しては打った場所の X Y あと passの三種類


###　実装すること

#### とりあえず予測(over fitting するくらいがいい)

- initialization
- forward propagation
- cost function
- back propagation


#### over fitting の解消
- regularization 
  - L2 regularization (add lambda) 
  - dropout regularization(Inverted dropout) add keep-prb
  - (Early stopping)


## neural network の検証

- train/dev/(test) で検証
- high bias/high varianceの表示

## outputディレクトリに検証結果を保存
保存するもの

- レイヤーの数
- それぞれのレイヤーのニューロンの数
- それぞれのレイヤーで使ったアクティベーション関数
- ラーニングレイトαの値
- trainデータの損失
- dev/(test)のデータの損失
- regularization のラムダの値
