from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
print(train.columns)
test = pd.read_csv('test.csv')
data = pd.concat([train, test], sort=False)

# missing values
median_imputer = SimpleImputer(strategy='median')
constant_imputer_cat = SimpleImputer(strategy='constant', fill_value='MISS')
data[['Fare']] = median_imputer.fit_transform(data[['Fare']])
data[['Embarked']] = constant_imputer_cat.fit_transform(data[['Embarked']])

# one-hot encoding
ohe_embarked = pd.get_dummies(data['Embarked'], prefix='ohe_embarked')
data = pd.concat([data, ohe_embarked], axis=1)
print(data.columns)
print(data.info())

# Label Encoder is for binary features
le = LabelEncoder()
data['encoded_sex'] = le.fit_transform(data['Sex'])

new_train = data[data['PassengerId'].isin(train['PassengerId'])]
new_test = data[data['PassengerId'].isin(test['PassengerId'])]
new_train = new_train.astype({'Survived': 'int64'})
new_test = new_test.drop(labels='Survived', axis=1)

X_train = new_train[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'encoded_sex', 'ohe_embarked_C', 'ohe_embarked_S', 'ohe_embarked_Q', 'ohe_embarked_MISS']]
X_test = new_test[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'encoded_sex', 'ohe_embarked_C', 'ohe_embarked_S', 'ohe_embarked_Q', 'ohe_embarked_MISS']]
y_train = new_train['Survived']

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
new_test['Survived'] = logreg.predict(X_test)

sample_submission = pd.read_csv('gender_submission.csv')
print(sample_submission.head())
submission = new_test[['PassengerId', 'Survived']]
submission.to_csv('3_sub.csv', index=False)
