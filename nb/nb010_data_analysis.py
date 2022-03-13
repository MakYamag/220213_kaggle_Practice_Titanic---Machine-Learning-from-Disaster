#!/usr/bin/env python
# coding: utf-8

# # Overview
# - *Name*を用いて同じ家族を同定し、生存率を計算した新たな列*Family_SurvRate*を作成する。
# - 

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


# In[2]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/train.csv')   # Google Colabの場合はこちら
data_train = pd.read_csv('../data/train.csv')   # ローカルの場合はこちら
data_train.head()


# In[3]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/test.csv')   # Google Colabの場合はこちら
data_test = pd.read_csv('../data/test.csv')   # ローカルの場合はこちら
data_test.head()


# ### データ前処理

# In[97]:


# 'TrainFlag'列追加
data_train['TrainFlag'] = True
data_test['TrainFlag'] = False

# 訓練用とテスト用のデータ結合
df = pd.concat([data_train, data_test])
df.index = df['PassengerId']
df = df.drop("PassengerId", axis = 1)

# =========
# 欠損値処理
# =========

# 'Embarked'
# -----------
# 欠損値は2つだけなので、最頻値('S')で埋めることとする
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

# 'Age'
# -----
# 'Age'は'Pclass'と相関が高いため、'Pclass'と'Sex'でグループ分けし、各グループの中央値で置き換える
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))

# 'Cabin'
# -------
# 一文字目を取り出して新たな列'Deck'を作成、欠損値はZで置き換え
df['Deck'] = df['Cabin'].apply(lambda d: d[0] if pd.notnull(d) else 'Z')

# 'Fare'
# ------
# testデータに1つだけあるので、平均値で埋める
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# =========
# 特徴量生成
# =========

# 'Name'
# ------
# 'Mr'などのタイトルを抜き出して新たな列'Title'を作成
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# 'Master'、'Miss'、'Mr'、'Mrs'に統合もしくはその他('Others')とする
df['Title'] = df['Title'].replace(['Mlle'], 'Miss')
df['Title'] = df['Title'].replace(['Countess', 'Mme', 'Lady'], 'Mrs')
df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Ms', 'Rev', 'Sir'], 'Others')

# 'Ticket'
# --------
# 'Ticket'の1文字目を抽出して新たな列'Ticket_first'作成
df['Ticket_first'] = df['Ticket'].apply(lambda t: str(t)[0])

# 'Ticket'の長さによる新たな列'Ticket_length'作成
df['Ticket_length'] = df['Ticket'].apply(lambda t: len(str(t)))

# 'Family_size'
# -------------
# 'SibSp'+'Parch'+1を新たな列'Family_size'に出力
df['Family_size'] = df['SibSp'] + df['Parch'] + 1

# 'Family_SurvRate': 同じ姓(Surname)を持つ家族内の生存率
# -------------------------------------------------------
# Surnameを取り出す関数extract_surnameを定義
def extract_surname(data):
    families = []
    
    for i in range (len(data)):
        name = data.iloc[i]
        
        # ()が付いている場合は、(の前までを取り出す
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
        
        # カンマ(,)の前までを姓として取り出す
        surname = name_no_bracket.split(',')[0]
        
        # 句読点をスペースに置き換えてスペースを削除する
        for p in string.punctuation:
            surname.replace(p, ' ').strip()
        
        families.append(surname)
    return families

# 'Surname'列を追加
df['Surname'] = extract_surname(df['Name'])

# 一時的にtrainを分離
df_temp_train = df[df['TrainFlag']==1]

# Surnameでグループ分けしたDataFrameで'SurvivalRate'を計算
df_temp_train_surname = pd.DataFrame(df_temp_train.groupby('Surname')['Survived'].sum())
df_temp_train_surname['FamilySize'] = df_temp_train.groupby('Surname')['Surname'].count()
df_temp_train_surname['SurvivalRate'] = df_temp_train_surname['Survived'] / df_temp_train_surname['FamilySize']

# surname_dictに辞書形式で出力し、新たな列'Family_SurvRate'にmap関数で展開する
surname_dict = df_temp_train_surname['SurvivalRate'].to_dict()
df['Family_SurvRate'] = df['Surname']
df['Family_SurvRate'] = df['Family_SurvRate'].map(surname_dict)

# 'Family_SurvRate'がNaN（Testデータのみに固有のSurname）については、平均で補完して'Family_SurvRate_NA'列に1を入れる
df['Family_SurvRate_NA'] = 0
df['Family_SurvRate_NA'].loc[df['Family_SurvRate'].isnull()==True] = 1
df['Family_SurvRate'] = df['Family_SurvRate'].fillna(df['Family_SurvRate'].mean())


# =========
# 特徴量整理
# =========

# 'Sex'、'Embarked'、'Deck'、'Title'、'Ticket_first'をone-hot-encodeする
df_oh = pd.get_dummies(df[['Sex', 'Embarked', 'Deck', 'Title', 'Ticket_first']], drop_first=True)

# one-hot-encodeデータを結合する
df_added = pd.concat([df, df_oh], axis=1)

# 'Name'、'Sex'、'Ticket'、'Cabin'、'Embarked'、'Deck'、'Title'、'Ticket_first'、'Surname'を削除する
df_deleted = df_added.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Deck', 'Title', 'Ticket_first', 'Surname'], axis=1)
df_deleted.info()

# 訓練用とテスト用に再び分割
df_train = df_deleted[df_deleted['TrainFlag']==True].drop(['TrainFlag'], axis=1)
df_test = df_deleted[df_deleted['TrainFlag']==False].drop(['TrainFlag'], axis=1)

# X、yとしてNumpy配列にする
X_train = df_train.drop(['Survived'], axis=1).values
y_train = df_train['Survived'].values
X_test = df_test.drop(['Survived'], axis=1).values

print('\nX_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)


# ### データ解析

# In[95]:


# 訓練用、テスト用にデータ分割する
from sklearn.model_selection import train_test_split

X_train, X_train_test, y_train, y_train_test = train_test_split(X_train, y_train, test_size=0.2,
                                                                            random_state=21, stratify=y_train)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_train_test: [0 1] =', np.bincount(y_train_test.astype(np.int64)))


# In[96]:


# =================================
# Pipeline: pl_scv
# SVC / k分割交差検証 / グリッドサーチ
# =================================

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pl_svc = make_pipeline(StandardScaler(), SVC(random_state=21, max_iter=5000))

from sklearn.model_selection import GridSearchCV

svc_param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['linear']},
                  {'svc__C': svc_param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': svc_param_range}]
svc_gs = GridSearchCV(estimator=pl_svc, param_grid=svc_param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
svc_gs.fit(X_train, y_train)

print('CV best accuracy:', svc_gs.best_score_)
print('Best parameters:', svc_gs.best_params_)
svc_bestclf = svc_gs.best_estimator_
print('Test accuracy: %f' % svc_bestclf.score(X_train_test, y_train_test))

svc_pred = svc_bestclf.predict(X_test)


# In[98]:


# =============================================
# Pipeline: pl_randf
# ランダムフォレスト / k分割交差検証 / グリッドサーチ
# =============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

pl_randf = make_pipeline(RandomForestClassifier(random_state=21, n_jobs=-1))

from sklearn.model_selection import GridSearchCV

rf_param_estimators_range = [100, 200, 300, 400, 500]
rf_param_depth_range = [5, 10, 15, 20, 25, 30]
rf_param_split_range = [5, 10, 15, 20, 25, 30]
rf_param_grid = [{'randomforestclassifier__criterion': ['gini', 'entropy'],
                  'randomforestclassifier__n_estimators': rf_param_estimators_range,
                  'randomforestclassifier__max_depth': rf_param_depth_range,
                  'randomforestclassifier__min_samples_split': rf_param_split_range}]
rf_gs = GridSearchCV(estimator=pl_randf, param_grid=rf_param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
rf_gs.fit(X_train, y_train)

print('CV best accuracy:', rf_gs.best_score_)
print(rf_gs.best_params_)
rf_bestclf = rf_gs.best_estimator_
print('Test accuracy: %f' % rf_bestclf.score(X_train_test, y_train_test))

rf_pred = rf_bestclf.predict(X_test)


# In[99]:


# =============================================
# Pipeline: pl_ada
# ADA Boost / k分割交差検証 / グリッドサーチ
# =============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline

pl_ada = make_pipeline(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=21), random_state=21))
ada_param_grid = [{'adaboostclassifier__base_estimator__criterion': ['gini', 'entropy'],
                   'adaboostclassifier__base_estimator__max_depth': [1, 5, 10, 15, 20],
                   'adaboostclassifier__base_estimator__min_samples_split': [1, 5, 10, 15, 20],
                   'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
                   'adaboostclassifier__n_estimators': [1, 5, 10, 15, 20, 25, 30],
                   'adaboostclassifier__learning_rate': [0.001, 0.01, 0.1, 1, 10, 100]}]
ada_gs = GridSearchCV(estimator=pl_ada, param_grid=ada_param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
ada_gs.fit(X_train, y_train)

print('CV best accuracy:', ada_gs.best_score_)
print(ada_gs.best_params_)
ada_bestclf = ada_gs.best_estimator_
print('Test accuracy: %f' % ada_bestclf.score(X_train_test, y_train_test))

ada_pred = ada_bestclf.predict(X_test)


# In[100]:


# ========================
# 多数決アンサンブルモデル
# ========================

from sklearn.ensemble import VotingClassifier

vote_clf = VotingClassifier(estimators=[('svc', svc_bestclf), ('rf', rf_bestclf), ('ada', ada_bestclf)])
vote_clf.fit(X_train, y_train)

print('Test accuracy: %f' % vote_clf.score(X_train_test, y_train_test))

vote_pred = vote_clf.predict(X_test)

