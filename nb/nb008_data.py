#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb007までは取り除いていた*Name*、*Ticket*、*Cabin*を特徴量として取り扱えるよう、データ処理を行う。

# In[238]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[255]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/train.csv')   # Google Colabの場合はこちら
df_train = pd.read_csv('../data/train.csv')   # ローカルの場合はこちら
df_train.head()


# In[256]:


print(df_train.isnull().sum())


# In[257]:


# 'Embarked'の欠損値処理
# =======================

print('Before: \n%s\n' % df_train['Embarked'].value_counts())

# 欠損値は2つだけなので、最頻値('S')で埋めることとする
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0])

print('After: \n%s\n' % df_train['Embarked'].value_counts())
print(df_train.isnull().sum())


# In[258]:


# 'Age'の欠損値処理
# ==================

print(df_train.corrwith(df_train['Age']), '\n')

# 'Age'は'Pclass'と相関が高いため、'Pclass'と'Sex'でグループ分けし、各グループの中央値で置き換える
print(df_train.groupby(['Pclass', 'Sex'])['Age'].median(), '\n')
df_train['Age'] = df_train.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
print(df_train.isnull().sum())


# In[90]:


# 特徴量をX,ラベルをyとして分離しNumpy配列にする
X = df_train_raw.drop(['Survived'], axis=1).values
y = df_train_raw['Survived'].values
print(X, '\n')

# 訓練用、テスト用にデータ分割する   # 本当は最初にする必要あり？
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))


# In[21]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
train_data = train_data_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
train_data_columns = train_data.columns.values


# In[34]:


# Pipeline: pl_scv
# 欠損値平均補完 / SVC
# =====================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pl_svc = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                       StandardScaler(), SVC(random_state=21, max_iter=10000))

from sklearn.model_selection import GridSearchCV
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': param_range}]
gs = GridSearchCV(estimator=pl_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[38]:


bestclf = gs.best_estimator_
print('Test accuracy: %.3f' % bestclf.score(X_test, y_test))

