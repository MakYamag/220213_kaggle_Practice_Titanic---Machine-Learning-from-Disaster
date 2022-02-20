#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb004までの欠損値平均補完では、訓練データ/テストデータ分割前に平均値補完していたので（テストデータ情報流出）、先に分割するように修正。
# - その他モデルはnb004ベース。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data_raw = pd.read_csv('../data/train.csv')
train_data_raw.head()


# In[29]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
train_data = train_data_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
train_data_columns = train_data.columns.values

# 特徴量をX、ラベルをyとして分離する
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']
X


# In[50]:


# 欠損値平均補完データの作成: train_data_imputed
# =============================================

# 訓練用、テスト用にデータ分割する   !!!テストデータ情報の混入防止!!!
from sklearn.model_selection import train_test_split
X_train, X_test, y_imp_train, y_imp_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))

# 欠損値を平均で補完   !!!テストデータにも訓練データでのfit情報を使う!!!
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(X_train)
X_imp_train = imr.transform(X_train)
X_imp_test = imr.transform(X_test)
#X_imp_train

# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_imp_train)
X_imp_train_std = sc.transform(X_imp_train)
X_imp_test_std = sc.transform(X_imp_test)
#X_imp_train_std


# In[65]:


# 欠損値行削除データの作成: train_data_dropna
# ============================================

# 欠損値行の削除
train_data_dropna = train_data.dropna(axis=0)

# ラベルを分離する: X_dna, y_dna
X_dna = train_data_dropna.drop(['Survived'], axis=1)
y_dna = train_data_dropna['Survived']
print(X_dna.shape)   # X.shape = (891, 8)

# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split
X_dna_train, X_dna_test, y_dna_train, y_dna_test = train_test_split(X_dna, y_dna, test_size=0.3, random_state=21, stratify=y_dna)

print('Label counts in y_dna: [0 1] =', np.bincount(y_dna))
print('Label counts in y_dna_train: [0 1] =', np.bincount(y_dna_train))
print('Label counts in y_dna_test: [0 1] =', np.bincount(y_dna_test))

# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_dna_train)
X_dna_train_std = sc.transform(X_dna_train)
X_dna_test_std = sc.transform(X_dna_test)
#X_dna_train_std


# In[82]:


# 決定木で分類モデル作成
# (tree_1: 欠損値平均補完データ使用)
# ==================================

from sklearn.tree import DecisionTreeClassifier
tree_1 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=21)
tree_1.fit(X_imp_train, y_imp_train)   # 決定木なので標準化データは使わない

# X_imp_testで分類予測
y_pred_1 = tree_1.predict(X_imp_test)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_imp_test, y_pred_1))


# 特徴量重要度を出力
tree_importance = tree_1.feature_importances_
tree_indices = np.argsort(tree_importance)[::-1]
tree_columns = X.columns
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f+1, 15, tree_columns[tree_indices[f]], tree_importance[tree_indices[f]]))


# In[37]:


# 決定木を図で出力(graphviz)
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data_1 = export_graphviz(tree_1, filled=True, rounded=True, class_names=['Dead', 'Survive'],
                             feature_names=X_imp.columns.values, out_file=None)
graph_1 = graph_from_dot_data(dot_data)
graph_1.write_png('tree_1.png')

from PIL import Image
im = Image.open("tree_1.png")
im_list = np.asarray(im)   # 画像をarrayに変換
plt.imshow(im_list)
plt.show()


# In[38]:


# 決定木を図で出力(dtreeviz)
from dtreeviz.trees import dtreeviz
viz = dtreeviz(tree_1, X_imp_train, y_imp_train, target_name='class', feature_names=X_imp.columns.values,
               class_names=['Dead', 'Survive'])
viz.save('tree_1_viz.svg')
viz.view()


# In[54]:


# 決定木で分類モデル作成
# (tree_2: 欠損値削除データ使用)
# ==============================

from sklearn.tree import DecisionTreeClassifier
tree_2 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=21)
tree_2.fit(X_dna_train, y_dna_train)   # 決定木なので標準化データは使わない

# X_dna_testで分類予測
y_pred_2 = tree_2.predict(X_dna_test)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_dna_test, y_pred_2))


# In[40]:


# 決定木を図で出力(dtreeviz)
from dtreeviz.trees import dtreeviz
viz = dtreeviz(tree_2, X_dna_train, y_dna_train, target_name='class', feature_names=X_dna.columns.values,
               class_names=['Dead', 'Survive'])
viz.save('tree_2_viz.svg')
viz.view()


# In[83]:


# ランダムフォレストで分類モデル作成
# (forest_1: 欠損値平均補完データ使用)
# ====================================
from sklearn.ensemble import RandomForestClassifier
forest_1 = RandomForestClassifier(criterion='gini', n_estimators=50, random_state=21, n_jobs=2)
forest_1.fit(X_imp_train, y_imp_train)   # 決定木なので標準化データは使わない

# X_imp_testで分類予測
y_for_pred_1 = forest_1.predict(X_imp_test)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_imp_test, y_for_pred_1))

# 特徴量重要度を出力
forest_importance = forest_1.feature_importances_
forest_indices = np.argsort(forest_importance)[::-1]
forest_columns = X.columns
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f+1, 15, forest_columns[forest_indices[f]], forest_importance[forest_indices[f]]))


# In[56]:


# ランダムフォレストで分類モデル作成
# (forest_2: 欠損値削除データ使用)
# ====================================
from sklearn.ensemble import RandomForestClassifier
forest_2 = RandomForestClassifier(criterion='gini', n_estimators=50, random_state=21, n_jobs=2)
forest_2.fit(X_dna_train, y_dna_train)   # 決定木なので標準化データは使わない

# X_dna_testで分類予測
y_for_pred_2 = forest_2.predict(X_dna_test)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_dna_test, y_for_pred_2))


# In[57]:


# KNN近傍法で分類モデル作成
# (knn_1: 欠損値平均補完データ使用)
# ==================================
from sklearn.neighbors import KNeighborsClassifier
knn_1 = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn_1.fit(X_imp_train_std, y_imp_train)

# X_imp_testで分類予測
y_knn_pred_1 = knn_1.predict(X_imp_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_imp_test, y_knn_pred_1))


# In[58]:


# KNN近傍法で分類モデル作成
# (knn_2: 欠損値削除データ使用)
# ==============================
from sklearn.neighbors import KNeighborsClassifier
knn_2 = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
knn_2.fit(X_dna_train_std, y_dna_train)

# X_dna_testで分類予測
y_knn_pred_2 = knn_2.predict(X_dna_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_dna_test, y_knn_pred_2))

