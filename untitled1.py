import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import random
import os
from	sklearn.model_selection	import	train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import timeit
from sklearn.metrics import roc_curve, auc
import AdirLib as adir
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from	sklearn.ensemble	import	RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score
os.chdir('C:/Users/user/Documents/Python/Titanic')




df=pd.read_csv('train.csv')
real=pd.read_csv('test.csv')

real['Survived']=np.nan

jdf=df.append(real).reset_index(drop=True)

def FillAgeNAN(df):

    df['hasAge']=list(map(np.isnan,df['Age']))
    df['title']=[a.split(',')[1].split(' ')[1] for a in df['Name']]
    
    
    for a in df[df['hasAge']==1]['title'].unique():
        valuetoput=df.groupby('title').mean()['Age'].sort_values()[a]
        df.loc[df['title']==a,'Age']=valuetoput
    return df



jdf=FillAgeNAN(jdf)


cols=['Survived','Pclass',  'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']

jdf.dropna(subset=['Embarked'],inplace=True,axis=0)
jdf.dropna(subset=['Fare'],inplace=True,axis=0)
# jdf.drop(columns=['Survived'],inplace=True)
jdf.reset_index(drop=True,inplace=True)

realpass=jdf[jdf['Survived'].isnull()].reset_index()
jdf=jdf[cols].copy()

df_cat=jdf[['Sex','Embarked']]
df_num=jdf[['Pclass', 'Age', 'SibSp','Parch', 'Fare','Survived']].copy()






onh = OneHotEncoder(drop='first').fit(df_cat)
df_cat_trans=pd.DataFrame(onh.transform(df_cat).toarray())



df_trans_j=df_cat_trans.join(df_num)


df_train=df_trans_j[-df_trans_j['Survived'].isnull()]
df_test=df_trans_j[df_trans_j['Survived'].isnull()]

df_train_X=df_train.drop(columns=['Survived'])
df_train_y=df_train['Survived']

df_test_X=df_test.drop(columns=['Survived'])




# X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, random_state=0)
X_train=df_train_X
y_train=df_train_y


clf = LogisticRegression(max_iter=555).fit(X_train, y_train)
# # clf = SVC(probability=True).fit(X_train, y_train)
# # clf = RandomForestClassifier().fit(X_train, y_train)

# stop = timeit.default_timer()

# print('Time end clf is ready: ', stop - start,'next is AUC')  

ss=pd.DataFrame(clf.predict(df_test_X),columns=['Survived'])
ans=realpass.drop(columns=['Survived']).join(ss)
ans.to_csv('ans1.csv')
# ans=
# # # clf = LinearRegression().fit(X_train, y_train)
# # # clf = LogisticRegression(max_iter=3776).fit(X_train, y_train)
# # # clf = SVC( C=1).fit(X_train, y_train)
# # clf = RandomForestClassifier().fit(X_train, y_train)

# def myscorer(clf,X,y):
#     fpr_lr, tpr_lr, _ = roc_curve(y, clf.predict(X))
#     return auc(fpr_lr, tpr_lr)

# # cv_scores = cross_val_score(clf, X, y,scoring='recall')
    
# ans=clf.predict(X_test)
# print(precision_score(y_test, ans, average='weighted',labels=np.unique(ans)))
# print(precision_score(y_test,ans,average='weighted'))
# print(recall_score(y_test,ans,average='weighted'))
# # newcabon=clf.predict()
# # # # fpr_lr, tpr_lr, _ = roc_curve(y_test, linreg.decision_function(X_test))
# # # # print('linreg auc:',auc(fpr_lr, tpr_lr))
# # fpr_lr, tpr_lr, _ = roc_curve(y_test, clf.predict(X_test))
# # print('AUC:',auc(fpr_lr, tpr_lr))
# # # # fpr_lr, tpr_lr, _ = roc_curve(y_test, SVCclf.decision_function(X_test))
# # # # print('SVM auc:',auc(fpr_lr, tpr_lr))
