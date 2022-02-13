#!/usr/bin/env python
# coding: utf-8

# # Overview
# - パーセプトロンモデルによって分類を試みた。
# - *Name*、*Ticket*、*Cabin*は、ひとまず特徴量から抜いた。

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


train_data = pd.read_csv('../data/train.csv')
train_data.head()


# In[4]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
X = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
X = pd.get_dummies(X)
X = X.drop(['Sex_male', 'Embarked_S'], axis=1)
X


# In[5]:


y = train_data['Survived']
y


# In[6]:


# 欠損値を平均値で補完する
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(X)
X_imputed = imr.transform(X)
X_imputed.shape


# In[7]:


# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=21, stratify=y)
print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))


# In[8]:


# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std[0:5]


# In[12]:


# パーセプトロンで分類モデル作成
# ===============================

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.01, max_iter=100, random_state=21)
ppn.fit(X_train_std, y_train)

# X_testで分類予測
y_pred = ppn.predict(X_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# In[ ]:




