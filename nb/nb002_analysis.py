#!/usr/bin/env python
# coding: utf-8

# # Overview
# - ロジスティック回帰モデルによって分類を試みた。
# - nb001と同様、*Name*、*Ticket*、*Cabin*は、ひとまず特徴量から抜いた。
# - *lr_1*はnb001と同様に欠損値を平均で補完、*lr_2*は欠損値を持つデータ行を削除して訓練した。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv('../data/train.csv')
train_data.head()


# In[3]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
X = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
X = pd.get_dummies(X)
X = X.drop(['Sex_male', 'Embarked_S'], axis=1)
X


# In[40]:


y = train_data['Survived']
y = pd.DataFrame(y)
y


# In[39]:


# 欠損値を平均値で補完する
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(X)
X_imputed = imr.transform(X)
X_imputed.shape
#X_imputed[888, 1]
#np.mean(X_imputed[:, 1])


# In[27]:


# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_imputed, y, test_size=0.3, random_state=21, stratify=y)
print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train_1))
print('Label counts in y_test: [0 1] =', np.bincount(y_test_1))


# In[28]:


# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_1)
X_train_1_std = sc.transform(X_train_1)
X_test_1_std = sc.transform(X_test_1)

X_train_1_std[0:5]


# In[95]:


# ロジスティック回帰で分類モデル作成
# (lr_1: 欠損値データ平均補完ver.)
# ==================================

from sklearn.linear_model import LogisticRegression
lr_1 = LogisticRegression(C=0.1, random_state=21, solver='lbfgs', max_iter=100, multi_class='auto')
lr_1.fit(X_train_1_std, y_train_1)

# X_testで分類予測
y_pred_1 = lr_1.predict(X_test_1_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test_1, y_pred_1))


# In[86]:


# 欠損値を含む行を削除する
X_y = X.join(y)
X_y_dropna = X_y.dropna(axis=0)
X_dropna = X_y_dropna.drop('Survived', axis=1)
y_dropna = X_y_dropna['Survived']
X_dropna.shape   # X.shape = (891, 8)


# In[89]:


# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_dropna, y_dropna, test_size=0.3, random_state=21, stratify=y_dropna)
X_train_2


# In[90]:


# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_2)
X_train_2_std = sc.transform(X_train_2)
X_test_2_std = sc.transform(X_test_2)

X_train_2_std[0:5]


# In[94]:


# ロジスティック回帰で分類モデル作成
# (lr_2: 欠損値データ削除ver.)
# ==================================

from sklearn.linear_model import LogisticRegression
lr_2 = LogisticRegression(C=0.1, random_state=21, solver='lbfgs', max_iter=100, multi_class='auto')
lr_2.fit(X_train_1_std, y_train_1)

# X_testで分類予測
y_pred_2 = lr_2.predict(X_test_2_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test_2, y_pred_2))

