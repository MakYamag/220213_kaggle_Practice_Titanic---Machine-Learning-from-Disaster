#!/usr/bin/env python
# coding: utf-8

# # Overview
# - SVMモデルによって分類を試みた。
# - nb001, nb002と同様、*Name*、*Ticket*、*Cabin*は、ひとまず特徴量から抜いた。データも欠損値平均補完と、欠損値削除の両方を用意した。
# - *svc_1*はlinear SVC: 欠損値平均補完、*svc_2*はlinear SVC: 欠損値削除、*svc_3*はkernel SVC(rbf): 欠損値削除として訓練した。

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


train_data_raw = pd.read_csv('../data/train.csv')
train_data_raw.head()


# In[6]:


# Passengerid, Name, Ticket, Cabin列を除いた特徴量を取得
train_data = train_data_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Sex, Embarked列をone-hot encordし、それぞれ1列を削除する
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
train_data_columns = train_data.columns.values
train_data


# In[11]:


# 欠損値を平均値で補完する: train_data_imputed
# =============================================
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(train_data)
train_data_imputed = pd.DataFrame(imr.transform(train_data), columns=train_data_columns)

# ラベルを分離する: X_imp, y_imp
X_imp = train_data_imputed.drop(['Survived'], axis=1)
y_imp = train_data_imputed['Survived']
X_imp


# In[28]:


# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split
X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(X_imp, y_imp, test_size=0.3, random_state=21, stratify=y_imp)

print('Label counts in y_imp: [0 1] =', np.bincount(y_imp))
print('Label counts in y_imp_train: [0 1] =', np.bincount(y_imp_train))
print('Label counts in y_imp_test: [0 1] =', np.bincount(y_imp_test))

# 特徴量を標準化する
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_imp_train)
X_imp_train_std = sc.transform(X_imp_train)
X_imp_test_std = sc.transform(X_imp_test)


# In[23]:


# 欠損値を含む行を削除する: train_data_dropna
# ============================================
train_data_dropna = train_data.dropna(axis=0)

# ラベルを分離する: X_dna, y_dna
X_dna = train_data_dropna.drop(['Survived'], axis=1)
y_dna = train_data_dropna['Survived']
X_dna   # X.shape = (891, 8)


# In[29]:


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


# In[35]:


# linear SVCで分類モデル作成
# (svc_1: 欠損値平均補完データ使用)
# ==================================

from sklearn.svm import SVC
svc_1 = SVC(kernel='linear', C=0.1, random_state=21, max_iter=-1)
svc_1.fit(X_imp_train_std, y_imp_train)

# X_imp_test_stdで分類予測
y_pred_1 = svc_1.predict(X_imp_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_imp_test, y_pred_1))


# In[52]:


# linear SVCで分類モデル作成
# (svc_2: 欠損値削除データ使用)
# ==============================

from sklearn.svm import SVC
svc_2 = SVC(kernel='linear', C=0.1, random_state=21, max_iter=-1)
svc_2.fit(X_dna_train_std, y_dna_train)

# X_dna_test_stdで分類予測
y_pred_2 = svc_2.predict(X_dna_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_dna_test, y_pred_2))


# In[79]:


# kernel SVCで分類モデル作成
# (svc_3: 欠損値削除データ使用)
# ==============================

from sklearn.svm import SVC
svc_3 = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=21, max_iter=-1)
svc_3.fit(X_dna_train_std, y_dna_train)

# X_dna_test_stdで分類予測
y_pred_3 = svc_3.predict(X_dna_test_std)
# 分類の正解率を計算
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_dna_test, y_pred_3))

