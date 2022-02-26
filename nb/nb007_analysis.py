#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb004の結果が一番良かった、欠損値平均補完/ランダムフォレストでパイプライン*pl_1*作成。
# - 層化k分割交差検証を実施。また、訓練データ数を横軸にとったLearning Curve、ランダムフォレストの決定木数を横軸に取ったValidation Curveを作成。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data_raw = pd.read_csv('../data/train.csv')
train_data_raw.head()


# In[18]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
train_data = train_data_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
train_data_columns = train_data.columns.values

# 特徴量をX,ラベルをyとして分離しNumpy配列にする
X = train_data.drop(['Survived'], axis=1).values
y = train_data['Survived'].values
X


# In[15]:


# 訓練用、テスト用にデータ分割する   !!!テストデータ情報の混入防止!!!
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))


# In[ ]:


# Pipeline: pl_scv
# 欠損値平均補完 / SVC
# =====================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pl_svc = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                       StandardScaler(), SVC(random_state=21))

from sklearn.model_selection import GridSearchCV
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__': ['poly', 'rbf', 'sigmoid'], 'svc_gamma': param_range}]
gs = GridSearchCV(estimator=pl_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)

