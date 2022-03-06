#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb007までは取り除いていた*Name*、*Ticket*、*Cabin*を特徴量として取り扱えるよう、データ処理を行う。
# - *Age*は*Sex*と*Pclass*のグループごとに中央値で補完。*Embarked*は最頻値で補完。*Cabin*は先頭のアルファベットを抽出し、欠損値はZで補完。
# - *Name*からTitleを取り出し、Master、Miss、Mr、Mrs、Othersに分類。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/train.csv')   # Google Colabの場合はこちら
df_train = pd.read_csv('../data/train.csv')   # ローカルの場合はこちら
df_train.head()


# In[3]:


# 特徴量をX,ラベルをyとして分離する
df_train_X = df_train.drop(['Survived'], axis=1)
df_train_y = df_train['Survived']
print(df_train_X.isnull().sum(), '\n')
print('Nan in y: %d' % df_train_y.isnull().sum())


# ### 欠損値処理

# In[4]:


# 'Embarked'の欠損値処理
# =======================

print('Before: \n%s\n' % df_train_X['Embarked'].value_counts())

# 欠損値は2つだけなので、最頻値('S')で埋めることとする
df_train_X['Embarked'] = df_train_X['Embarked'].fillna(df_train_X['Embarked'].mode().iloc[0])

print('After: \n%s\n' % df_train_X['Embarked'].value_counts())
print(df_train_X.isnull().sum())


# In[5]:


# 'Age'の欠損値処理
# ==================

print(df_train.corrwith(df_train['Age']), '\n')

# 'Age'は'Pclass'と相関が高いため、'Pclass'と'Sex'でグループ分けし、各グループの中央値で置き換える
print(df_train_X.groupby(['Pclass', 'Sex'])['Age'].median(), '\n')
df_train_X['Age'] = df_train_X.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
print(df_train_X.isnull().sum())


# In[6]:


# 'Cabin'の欠損値処理
# ====================

df_train_X['Cabin'].unique()

# 一文字目を取り出して新たな列'Deck'を作成、欠損値はZで置き換え
df_train_X['Deck'] = df_train_X['Cabin'].apply(lambda d: d[0] if pd.notnull(d) else 'Z')
df_train_X['Deck'].unique()
print(df_train_X.isnull().sum())


# ### 特徴量生成

# In[7]:


# 'Name'の特徴量生成
# ===================

# 'Mr'などのタイトルを抜き出して新たな列'Title'を作成
df_train_X['Title'] = df_train_X['Name'].str.extract('([A-Za-z]+)\.', expand=False)
print(df_train_X.groupby(['Title'])['Name'].count(), '\n')

# 'Master'、'Miss'、'Mr'、'Mrs'に統合もしくはその他('Others')とする
df_train_X['Title'] = df_train_X['Title'].replace(['Mlle'], 'Miss')
df_train_X['Title'] = df_train_X['Title'].replace(['Countess', 'Mme', 'Lady'], 'Mrs')
df_train_X['Title'] = df_train_X['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Ms', 'Rev', 'Sir'], 'Others')

print(df_train_X['Title'].unique())


# In[8]:


# 'Ticket'の特徴量生成
# =====================

# 'Ticket'の1文字目を抽出して新たな列'Ticket_first'作成
df_train_X['Ticket_first'] = df_train_X['Ticket'].apply(lambda t: str(t)[0])

# 'Ticket'の長さによる新たな列'Ticket_length'作成
df_train_X['Ticket_length'] = df_train_X['Ticket'].apply(lambda t: len(str(t)))

df_train_X.head()


# In[9]:


# 'Family_size'の作成
# ====================

# 'SibSp'+'Parch'+1を新たな列'Family_size'に出力
df_train_X['Family_size'] = df_train_X['SibSp'] + df_train['Parch'] + 1

df_train_X.groupby(['Family_size'])['PassengerId'].count()


# ### 特徴量整理

# In[10]:


# 'Sex'、'Embarked'、'Deck'、'Title'、'Ticket_first'をone-hot-encodeする
df_train_X_oh = pd.get_dummies(df_train_X[['Sex', 'Embarked', 'Deck', 'Title', 'Ticket_first']], drop_first=True)

# one-hot-encodeデータを結合する
df_train_X_added = pd.concat([df_train_X, df_train_X_oh], axis=1)
print(df_train_X_added.shape)

# 'Name'、'Sex'、'Ticket'、'Cabin'、'Embarked'、'Deck'、'Title'、'Ticket_first'を削除する
df_train_X_deleted = df_train_X_added.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Deck', 'Title', 'Ticket_first'], axis=1)
print(df_train_X_deleted.shape, '\n')
df_train_X_deleted.info()

# X、yとしてNumpy配列にする
X = df_train_X_deleted.values
y = df_train_y.values


# ### データ解析

# In[11]:


# 訓練用、テスト用にデータ分割する   # 本当は最初にしたほうがいい
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y: [0 1] =', np.bincount(y))
print('Label counts in y_train: [0 1] =', np.bincount(y_train))
print('Label counts in y_test: [0 1] =', np.bincount(y_test))


# In[13]:


# Pipeline: pl_scv
# SVC / k分割交差検証 / グリッドサーチ
# =====================================

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pl_svc = make_pipeline(StandardScaler(), SVC(C=10.0, kernel='rbf', gamma=0.01, random_state=21, max_iter=5000))   # CVの最適値入力済

from sklearn.model_selection import GridSearchCV

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': param_range}]
gs = GridSearchCV(estimator=pl_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print('CV best accuracy:', gs.best_score_)
print(gs.best_params_)
bestclf = gs.best_estimator_
print('Test accuracy: %f' % bestclf.score(X_test, y_test))


# In[14]:


# Pipeline: pl_randf
# ランダムフォレスト / k分割交差検証 / グリッドサーチ
# ====================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

pl_randf = make_pipeline(RandomForestClassifier(criterion='entropy', n_estimators=400, 
                                                max_depth=8, random_state=21, n_jobs=2))   # CVの最適値入力済

from sklearn.model_selection import GridSearchCV

param_estimators_range = [100, 200, 300, 400, 500]
param_depth_range = [4, 6, 8, 10]
param_grid = [{'randomforestclassifier__criterion': ['gini', 'entropy'],
               'randomforestclassifier__n_estimators': param_estimators_range,
               'randomforestclassifier__max_depth': param_depth_range}]
gs = GridSearchCV(estimator=pl_randf, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print('CV best accuracy:', gs.best_score_)
print(gs.best_params_)
bestclf = gs.best_estimator_
print('Test accuracy: %f' % bestclf.score(X_test, y_test))


# In[22]:


# Learning Curve
# ===============

from sklearn.model_selection import learning_curve

fig = plt.figure(figsize=(9.6, 3.2), dpi=80)

# SVCのプロット
svc_train_sizes, svc_train_scores, svc_valid_scores = learning_curve(estimator=pl_svc, X=X_train, y=y_train,
                                                       train_sizes=np.linspace(0.1, 1, 10), cv=10, n_jobs=-1)
svc_train_mean = np.mean(svc_train_scores, axis=1)
svc_train_std = np.std(svc_train_scores, axis=1)
svc_valid_mean = np.mean(svc_valid_scores, axis=1)
svc_valid_std = np.std(svc_valid_scores, axis=1)

ax1 = fig.add_subplot(1, 2, 1)
plt.plot(svc_train_sizes, svc_train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(svc_train_sizes, svc_train_mean+svc_train_std, svc_train_mean-svc_train_std, color='blue', alpha=0.2)
plt.plot(svc_train_sizes, svc_valid_mean, color='green', marker='s', markersize=5, linestyle='--', label='Validation accuracy')
plt.fill_between(svc_train_sizes, svc_valid_mean+svc_valid_std, svc_valid_mean-svc_valid_std, color='green', alpha=0.2)
plt.grid()
plt.title('SVC')
plt.ylim(0.7, 1.0)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend()

# ランダムフォレストのプロット
randf_train_sizes, randf_train_scores, randf_valid_scores = learning_curve(estimator=pl_randf, X=X_train, y=y_train,
                                                       train_sizes=np.linspace(0.1, 1, 10), cv=10, n_jobs=-1)
randf_train_mean = np.mean(randf_train_scores, axis=1)
randf_train_std = np.std(randf_train_scores, axis=1)
randf_valid_mean = np.mean(randf_valid_scores, axis=1)
randf_valid_std = np.std(randf_valid_scores, axis=1)

ax2 = fig.add_subplot(1, 2, 2)
plt.plot(randf_train_sizes, randf_train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(randf_train_sizes, randf_train_mean+randf_train_std, randf_train_mean-randf_train_std, color='blue', alpha=0.2)
plt.plot(randf_train_sizes, randf_valid_mean, color='green', marker='s', markersize=5, linestyle='--', label='Validation accuracy')
plt.fill_between(randf_train_sizes, randf_valid_mean+randf_valid_std, randf_valid_mean-randf_valid_std, color='green', alpha=0.2)
plt.grid()
plt.title('Random Forest Classifier')
plt.ylim(0.7, 1.0)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend()

# プロット画像保存
plt.savefig('../image/nb008_learningcurve.png')

