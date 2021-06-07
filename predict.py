# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
import sweetviz as sv
data = pd.read_csv("Training Data.csv")
df=data[['clientid_cr','pubclientid_cr','ip_cr','clmbuserid_cr','imprid_cr','siteId_cr','goalid_cr',
'cityId_cr','stateId_cr','countryDimId_cr','browserId_cr','adslotdimid_cr','itemid_cr','ispDimId_cr',
'deviceId_cr','osVerDimId_cr','conversion_fraud'
]]

my_report = sv.analyze(df)
my_report.show_html() 

#print(df.isnull().sum())
df = df.drop(['deviceId_cr'],axis=1)

corr = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,cmap='rainbow',annot=True)
plt.show()

df=df.drop(['countryDimId_cr'],axis=1)

op=df.conversion_fraud
ip=df.drop(['conversion_fraud'],axis=1)

s = (ip.dtypes == 'object')
categorical_cols = list(s[s].index)

print("Categorical variables:")
print(categorical_cols)

n = (ip.dtypes != 'object')
numerical_cols = list(n[n].index)
print("numerical_cols")
print(numerical_cols)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])




#model = LogisticRegression(solver='lbfgs', max_iter=1000)


#model = XGBRegressor(n_estimators=2000, learning_rate=0.07, n_jobs=4)

model= RandomForestClassifier(n_estimators=100)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(ip,op, test_size=0.2)

my_pipeline.fit(x_train, y_train)
preds = my_pipeline.predict(x_valid)

print(my_pipeline.score(x_valid,y_valid))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_valid, preds))

test_df=pd.read_csv('Test Data.csv')
test_ip = test_df.drop(['record_id'],axis=1)
test_pred=my_pipeline.predict(test_ip)



submission = pd.DataFrame({
        "record_id": test_df['record_id'],
        "conversion_fraud": test_pred
    })
submission.to_csv('submission.csv', index=False)



 