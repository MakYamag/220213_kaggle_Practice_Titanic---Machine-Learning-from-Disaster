#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb004の結果が一番良かった、欠損値平均補完/ランダムフォレストでパイプライン*pl_1*作成。
# - 層化k分割交差検証を実施。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data_raw = pd.read_csv('../data/train.csv')
train_data_raw.head()


# In[30]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
train_data = train_data_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
train_data_columns = train_data.columns.values

# 特徴量をX,ラベルをyとして分離しNumpy配列にする
X = train_data.drop(['Survived'], axis=1).values
y = train_data['Survived'].values
X


# In[31]:


# 訓練用、テスト用にデータ分割する   !!!テストデータ情報の混入防止!!!
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))


# In[32]:


# Pipeline: pl_1
# 欠損値平均補完 / ランダムフォレスト
# ===================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

pl_1 = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                     StandardScaler(),
                     RandomForestClassifier(criterion='gini', n_estimators=50, random_state=21, n_jobs=2))
pl_1.fit(X_train, y_train)
print('Accuracy: %.3f' % pl_1.score(X_test, y_test))


# In[33]:


# Stratified k-fold cross validation
# ===================================

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []

for k, (train, test) in enumerate(kfold):
    pl_1.fit(X_train[train], y_train[train])
    score = pl_1.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %d, Class dist: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train[train]), score))

