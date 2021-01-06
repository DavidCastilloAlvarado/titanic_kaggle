import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import random
import time
from tqdm import tqdm
import IPython.display as ipd
import seaborn as sns
import itertools
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier as KNC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import collections

def pre_process_dataset(path_test='dataset/test.csv',path_train='dataset/train.csv',all_categorical=False, Test=True, fillna_age = 'mean'):
    """
    fillna_age = 'mean' para rellenar con la media total
                 'None' para rellenar con la media agrupada según sexo y clase
    """  
    DB_train = pd.read_csv(path_train)
    DB_test = pd.read_csv(path_test) if Test else DB_train
    
    DB_test_orig = DB_test.copy()
    embarked_mode = DB_train['Embarked'].mode()[0]
    age_main = DB_train['Age'].mean()
    fare_main = DB_train['Fare'].mean()
    # Media Age for Pclass, sex and Embarked
    media_age = DB_train.dropna(subset=['Age'])[['Age','Pclass','Sex','Embarked']].groupby(['Pclass', 'Sex','Embarked'], as_index=False ).mean().sort_values(by='Age', ascending=True)
    #media_age = DB_train.dropna(subset=['Age'])[['Age','Pclass','Sex']].groupby(['Pclass', 'Sex'], as_index=False ).mean().sort_values(by='Age', ascending=True)
    def impute_years(x):
        if x['Age'] == x['Age'] :
            return x['Age']
        else:
            return media_age.loc[media_age['Pclass']==x['Pclass'] , ['Age', 'Sex','Embarked']].loc[media_age['Sex']==x['Sex'] , ['Age','Embarked']].loc[media_age['Embarked']==x['Embarked'] , ['Age']]['Age'].tolist()[0]
#             return media_age.loc[media_age['Pclass']==x['Pclass'] , ['Age', 'Sex']].loc[media_age['Sex']==x['Sex'] , ['Age']]['Age'].tolist()[0]
    
    ageband_list = [min(0,DB_train['Age'].min()),16,32,48,64,max(100,DB_train['Age'].max())]
    fareband_lost = [min(-1,DB_train['Fare'].min()-1),85,170,256,426,max(600,DB_train['Fare'].max())]

    ## Fusion SibSp abd Parch
    DB_test['n_parents'] = DB_test['SibSp'] + DB_test['Parch']
    DB_test['accompanied'] = DB_test['n_parents'].apply(lambda x: x if x <7 else 'nn')
    # Count how many cabins a passenger bought
    DB_test['Count_Cabin'] = DB_test['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    # The passenger bought a cabin yes or not
    DB_test['b_Cabin'] = DB_test['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    # Extracting titles from Name
    DB_test['title_Name'] = DB_test['Name'].apply(lambda x: x.split(',')[1:][0].split('.')[0].strip())
    # Grouping title names using only the predominants
    title_pred = ['Mr', 'Miss', 'Mrs','Master']
    DB_test['title_Name'] = DB_test['title_Name'].apply(lambda x: x if x in title_pred else 'Others')
    # Fill NAN values
    DB_test['Embarked'] = DB_test['Embarked'].fillna(embarked_mode)
    # 86 values are missed - Fill nan Age
    DB_test['Age'] = DB_test['Age'].fillna(age_main) if fillna_age == 'mean' else DB_test.apply(impute_years, axis=1) 
    ## Creating age band
    DB_test['AgeBand'] = pd.cut(DB_test['Age'], ageband_list)
    ## Fare fill nan Fare
    DB_test['Fare'] = DB_test['Fare'].fillna(fare_main) # in test set only one is missed
    ## Creating fare band
    DB_test['FareBand'] = pd.cut(DB_test['Fare'], fareband_lost)
    # Normlalizing data 
    DB_test['norm_Fare']= DB_test['Fare'].apply(lambda x : np.log10(x+1))
    DB_test['norm_Age']= DB_test['Age'].apply(lambda x : x)
    DB_test['Sex'] = DB_test['Sex'].apply(lambda x: 1 if x == "male" else 0)
    if all_categorical:
        DB_test.index  = DB_test.PassengerId
        DB_test = DB_test.drop(columns=[ 'Cabin','Fare','Name','PassengerId', 'Count_Cabin','n_parents','Ticket','SibSp','Parch','norm_Fare', 'norm_Age','Age'])
        DB_test['Pclass'] = DB_test['Pclass'].astype(str) # categorical feature
        DB_test['accompanied'] = DB_test['accompanied'].astype(str) # categorical feature
        #DB_test['b_Cabin'] = DB_test['b_Cabin'].astype(str) # categorical feature
        DB_test['AgeBand'] = DB_test['AgeBand'].astype(str) # categorical feature
        DB_test['AgeBand'] = DB_test['AgeBand'].apply(lambda x : x.replace('(','').replace(']','').replace(',','_'))
        DB_test['FareBand'] = DB_test['FareBand'].astype(str) # categorical feature
        DB_test['FareBand'] = DB_test['FareBand'].apply(lambda x : x.replace('(','').replace(']','').replace(',','_'))
        DB_test.info()
        DB_test.head()
    else:
        DB_test.index  = DB_test.PassengerId
        DB_test= DB_test.drop(columns=[ 'Cabin','Fare','Name','PassengerId', 'Age','Count_Cabin','n_parents','Ticket','SibSp','Parch','AgeBand','FareBand' ])
        DB_test['Pclass'] = DB_test['Pclass'].astype(str) # categorical feature
        DB_test['accompanied'] = DB_test['accompanied'].astype(str) # categorical feature
        #DB_test['b_Cabin'] = DB_test['b_Cabin'].astype(str) # categorical feature
        DB_test.info()
        DB_test.head()
    DB_test = pd.get_dummies(DB_test)
    if Test :
        return DB_test,DB_test_orig 
    else:
        return DB_test.drop(columns=['Survived']), DB_test['Survived']
        
    
    
def balancingClasses_Smoteenn(x_train, y_train,random_state):
    
    # Using SMOTEEN to balance our training data points
    smn = SMOTEENN(random_state=random_state)
    features_balanced, target_balanced = smn.fit_resample(x_train, y_train)
    
    print("Count for each class value after SMOTEEN:", collections.Counter(target_balanced))
    
    return features_balanced, target_balanced


def balancingClasses_Smote(x_train, y_train,random_state):

    # Using SMOTE to to balance our training data points
    sm = SMOTE(random_state=random_state)
    features_balanced, target_balanced = sm.fit_resample(x_train, y_train)

    print("Count for each class value after SMOTE:", collections.Counter(target_balanced))

    return features_balanced, target_balanced