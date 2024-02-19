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
from	sklearn.metrics	import	confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from	sklearn.preprocessing	import	PolynomialFeatures


start = timeit.default_timer()

df=pd.read_csv('train.csv')
real=pd.read_csv('test.csv')
df1=pd.read_csv('train.csv')



# real['type'], df['type']='test','train'


# real['Survived']=np.nan

# jdf=df.append(real).reset_index(drop=True)

def setSex(df):
    df['Sex']=np.where(df['Sex']=='female',1,0)
    return df

def dropCabin(df):
    if 'Cabin' not in df.columns: return df
    df.drop(columns=['Cabin'],inplace=True)
    return df

def FillAgeNAN(dfc):
    df=dfc.copy()
    df['hasAge']=list(map(np.isnan,df['Age']))
    df['title']=[a.split(',')[1].split(' ')[1] for a in df['Name']]
    
    agedict={'Mr.':30, 'Mrs.':35, 'Miss.':22, 'Ms.':28, 'Master.':3,'Dr.':46.5}
    
    
    for a in df[df['hasAge']==1].index:
        # print(a)
        # valuetoputmedian=df.groupby('title').median()['Age'].sort_values()[a]
        # valuetoputmean=df.groupby('title').mean()['Age'].sort_values()[a]
        df.loc[a,'Age']=agedict[df.loc[a,'title']]
    return df.drop(columns=['hasAge','title'])

def GetLastName(df):
    df['LastName']=[a.split(',')[0]+str(b) for a,b in df[['Name','Fare']].values]
    return df
# s=jdf.groupby('type').agg(func=lambda x: x.isna().sum()).astype('int')

def setFamilySize(df):
    df['FamilySize']=[a+b+1 for a,b in df[['Parch','SibSp']].values]
    return df

def setEmbarked(df):
    embarked_dic={'S':1,'C':2,'Q':3}
    df.dropna(subset=['Embarked'],inplace=True)
    df['Embarked']=[embarked_dic[x] for x in df['Embarked']]
    return df

def applyall(df):
    # df=GetLastName(df)    
    df=setSex(df)
    df=FillAgeNAN(df)
    df= dropCabin(df)
    # setFamilySize(df)
    setEmbarked(df)
    return df


df=applyall(df)
real=applyall(real)

# sns.catplot(x='title',y='Age',data=real, hue='hasAge')
# sns.catplot(x='title',y='Age',data=df2, hue='Survived')

# fig,ax = plt.subplots(3,1)

# sns.distplot(df2[df2['hasAge']==False]['Age'],ax=ax[0])
# sns.distplot(df2['Age'],ax=ax[1])
# sns.distplot(real['Age'],ax=ax[2])





y=df['Survived'].copy()
df2=df.copy()
df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'],inplace=True)
pid=real['PassengerId']
real.drop(columns=['PassengerId', 'Name', 'Ticket'],inplace=True)















df_dum=pd.get_dummies(df)
real_dum=pd.get_dummies(real)


poly = PolynomialFeatures(degree=2,order='C')
df_poly=poly.fit_transform(df_dum)

# onh = OneHotEncoder(drop='first').fit(df_cat)
# df_cat_trans=pd.DataFrame(onh.transform(df_cat).toarray())


X=df_poly


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# X_train=df_train_X
# y_train=df_train_y


# clf = LogisticRegression(max_iter=555,C=0.01).fit(X_train, y_train)
# clf = SVC()
# clf=SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False).fit(X_train, y_train)
clf = RandomForestClassifier().fit(X_train, y_train)



# # stop = timeit.default_timer()

# # print('Time end clf is ready: ', stop - start,'next is AUC')  

# ss=pd.DataFrame(clf.predict(df_test_X),columns=['Survived'])
# ans=realpass.drop(columns=['Survived']).join(ss)
# # ans=
# # # # clf = LinearRegression().fit(X_train, y_train)
# # # # clf = LogisticRegression(max_iter=3776).fit(X_train, y_train)
# # # # clf = SVC( C=1).fit(X_train, y_train)
# # # clf = RandomForestClassifier().fit(X_train, y_train)

# def myscorer(clf,X,y):

ypred=clf.predict(X_test)



print('naccuracy_score: {}'.format(accuracy_score(y_test,ypred)))

    
# cv_scores = cross_val_score(clf, df_dum, y,scoring='accuracy')
# print(cv_scores)

# ans=clf.predict(df_dum)
# confusion_matrix(y,ans)

'''
clf1 = RandomForestClassifier()#.fit(X_train, y_train)
parameters = {'n_estimators':[50,100,150],'max_depth':[1,2,3,4,5,6,7,None], 'min_samples_split':[2, 3,4,5]}
gclf = GridSearchCV(clf1, parameters)
gclf.fit(X_train, y_train)
best_gclf=gclf.best_estimator_

print('gclf.best_params_ = {}'.format(gclf.best_params_))
# print('gclf.best_params_ = {}\ngclf.best_score_ = {}'.format(gclf.best_params_,gclf.best_score_))
cross_val_score(best_gclf, df_dum, y,scoring='accuracy')
print('naccuracy_score: {}'.format(accuracy_score(y_test,best_gclf.predict(X_test))))
cv_scores = cross_val_score(best_gclf, df_dum, y,scoring='accuracy')
print(cv_scores)

print(pd.DataFrame([df_dum.columns,gclf.best_estimator_.feature_importances_]).T.sort_values(by=1))
'''








# ans=pd.DataFrame()
# ans['Survived']=best_gclf.predict(real_dum)
# ans['PassengerId']=pid
# ans.to_csv('ans3.csv',index=False)

#RandomSearchCV
'''
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)




ypred=rf_random.best_estimator_.predict(X_test)
print('naccuracy_score: {}'.format(accuracy_score(y_test,ypred)))



'''



stop = timeit.default_timer()
print('Time end clf is ready: ', stop - start,'next is AUC')  












