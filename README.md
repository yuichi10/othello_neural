# othello_neural

# やること
## neural networkの構築

5 layer くらいで最初やってみる

それぞれのニューロンの数

64 100 150 100 80 65 

input X はボードの状態
output Y はボードの打った場所を1にしたもの(passも含めて65通り)


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

#### 改良
- normalizing

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
