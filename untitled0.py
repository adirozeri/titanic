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



def flipstate(l):
 return list(map(lambda x:not(x),l))


start = timeit.default_timer()
print('Time end clf is ready: ', stop - start,'next is AUC')  
`

# originaldf_=pd.read_csv('train.csv')
os.chdir('C:/Users/user/Documents/Python/Titanic')
df=pd.read_csv('train.csv')



def FillAgeNAN(df):

    df['hasAge']=list(map(np.isnan,df['Age']))
    df['title']=[a.split(',')[1].split(' ')[1] for a in df['Name']]
    
    
    for a in df[df['hasAge']==1]['title'].unique():
        valuetoput=df.groupby('title').mean()['Age'].sort_values()[a]
        df.loc[df['title']==a,'Age']=valuetoput
    return df



asd=FillAgeNAN(df)

cols=['Pclass',  'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']
asd=asd[cols].copy()
asd.dropna(subset=['Embarked'],inplace=True,axis=0)


asd['hascabin']=asd['Cabin'].isnull()


aa=asd[['Sex','Embarked']].copy()
onh = OneHotEncoder(drop='first').fit(aa)
aa_trans=pd.DataFrame(onh.transform(aa).toarray())




         
aa_trans_join=aa_trans.join(df[['Pclass', 'Age', 'SibSp','Parch', 'Fare', 'Cabin']])
         
df_for_fit=aa_trans_join[-aa_trans_join['Cabin'].isnull()]


y=df_for_fit['Cabin'].copy()
X=df_for_fit.drop(columns=['Cabin'])



# X=train.drop(columns=['Cabin']).copy()
df_for_pred=aa_trans_join[aa_trans_join['Cabin'].isnull()]


##############X.y readu##################################



asd_trans=pd.DataFrame(onh.transform(asd.drop(['hascabin','Cabin'],axis=1)).toarray())

X_for_fit=asd_trans[asd['hascabin'].reset_index(drop=True)==True]
y_forfit = asd[asd['hascabin']==True]['Cabin']

X_for_reak=asd_trans[asd['hascabin'].reset_index(drop=True)==False]


# x_newCabin=pd.DataFrame(onh.transform(train).toarray())

le=LabelEncoder().fit(y_forfit)
y=le.transform(y_forfit)

X=X_for_fit

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



clf = LogisticRegression().fit(X, y)
# clf = SVC(probability=True).fit(X_train, y_train)
# clf = RandomForestClassifier().fit(X_train, y_train)

stop = timeit.default_timer()

print('Time end clf is ready: ', stop - start,'next is AUC')  


# clf.predict(X_test).to_csv('ans.csv')

# # clf = LinearRegression().fit(X_train, y_train)
# # clf = LogisticRegression(max_iter=3776).fit(X_train, y_train)
# # clf = SVC( C=1).fit(X_train, y_train)
# clf = RandomForestClassifier().fit(X_train, y_train)

def myscorer(clf,X,y):
    fpr_lr, tpr_lr, _ = roc_curve(y, clf.predict(X))
    return auc(fpr_lr, tpr_lr)

# cv_scores = cross_val_score(clf, X, y,scoring='recall')
    
ans=clf.predict(X_test)
print(precision_score(y_test, ans, average='weighted',labels=np.unique(ans)))
print(precision_score(y_test,ans,average='weighted'))
print(recall_score(y_test,ans,average='weighted'))
# newcabon=clf.predict()
# # # fpr_lr, tpr_lr, _ = roc_curve(y_test, linreg.decision_function(X_test))
# # # print('linreg auc:',auc(fpr_lr, tpr_lr))
# fpr_lr, tpr_lr, _ = roc_curve(y_test, clf.predict(X_test))
# print('AUC:',auc(fpr_lr, tpr_lr))
# # # fpr_lr, tpr_lr, _ = roc_curve(y_test, SVCclf.decision_function(X_test))
# # # print('SVM auc:',auc(fpr_lr, tpr_lr))
