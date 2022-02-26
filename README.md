# 220213_kaggle_Practice_Titanic---Machine-Learning-from-Disaster
かの有名なTitanicコンペ練習用レポジトリ。

## Log
### 220213
- ひとまずデータダウンロードし、欠損値補完、データ分割、標準化などやってみた後、パーセプトロンモデルで分類してみた。

#### [nb001]
- パーセプトロンモデルにて分類。正解率 0.746
- 欠損値は*Age*にNaNが含まれていたので、平均値で補完。
- 扱いが分からなかった*Name*、*Ticket*、*Cabin*はひとまず特徴量から抜いた。

<br><br>
### 220216
- ロジスティック回帰モデルで分類。

#### [nb002]
- nb001と同様に、欠損値（Age列）を平均値で補完したところ、正解率0.799。
- 欠損値のあるデータ行を削除してみたところ、データ数は891 ---> 714になったが、正解率は0.837に上昇。

<br><br>
### 220219
- SVM、決定木、ランダムフォレスト、K最近傍法にトライ。決定木をdtreevizで可視化するのに少し手間取ったが、普通にgraphvizで可視化するより全然わかりやすくてちょっと感動。
- *Name*、*Ticket*、*Cabin*は抜いたままでやっているが、分類正解率的にはどのアルゴリズムでも頭打ちな感はある。
- ハイパーパラメータはいくつか手動で試して適当によさげなのを選んでいるだけなので、そろそろパラメータ最適化も試さないといけない。

#### [nb003]
- linear SVCで欠損値補完：0.776、欠損値削除：0.805、kernel SVCで欠損値削除：0.823だった。

#### [nb004]
- 決定木で欠損値補完：0.828、欠損値削除：0.823だった。
- ランダムフォレストで欠損値補完：0.821、欠損値削除：0.786だった。
- K最近傍法で欠損値補完：0.825、欠損値削除：0.819だった。

<br><br>
### 220220
- 前日までのデータ前処理で、データ全体を使って欠損値平均補完してから訓練データ、テストデータに分割していたが、良くない（テストデータのリーク）があるとわかったので修正した。
- ついでに決定木、ランダムフォレストで特徴量の重要度（寄与度）を出力した。決定木は性別だけで約0.6だったが、ランダムフォレストでは運賃、年齢、性別がそれぞれ0.25~0.30前後と、違いがあって面白い。

#### [nb005]
- nb004ベースで、データ前処理のときに分割を先に持ってくるように修正した。
- ランダムフォレストの欠損値補完：0.821のみ0.836に上がったが、他は変わらなかった。リークしていたので修正後は下がると思ったが...？

<br><br>
### 220223
- データ前処理、モデルフィッティングまで一連の流れをパイプラインにした。
- 交差検証の足掛かりとして、k分割交差検証を実装した。

[nb006]
- nb004で一番結果の良かった欠損値平均補完/ランダムフォレストモデルでパイプラインを作成。
- 層化k分割交差検証ができるようにした。

<br><br>
### 220226
- Learning Curve、Validation Curveをプロット。
- グリッドサーチによるハイパーパラメータ探索を始めたが、組み合わせ数を多くしすぎて断念。続きは明日やる。

[nb006]
- k分割交差検証し、その結果をLearinig Curve（横軸が訓練データ数）、Validation Curve（横軸がランダムフォレストの決定木数）でプロットした。
- どちらのプロットでも、訓練データ数、決定木数に関わらず、訓練データ正解率0.97程度、検証データ正解率0.80程度と、バリアンスの高さが浮き彫りになった。データをしっかり見て、特徴量を増やす必要がある。

