from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print('Train shape:', train.shape)
#print('Train columns:', train.columns.tolist())
#print(train.Survived.value_counts())
#print(train.Pclass.value_counts())
#print(train.Name.value_counts())
#print(train.Sex.value_counts())
#print(train.Age.value_counts())
#print(train.SibSp.value_counts())
#print(train.Parch.value_counts())
#print(train.Ticket.value_counts())
#print(train.Fare.value_counts())
#print(train.Cabin.value_counts())
#print(train.Embarked.value_counts())
#print(train.describe())
#print(test.info())
#X_train = train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
X_train = train[['PassengerId', 'Pclass', 'SibSp', 'Parch']]
X_test = test[['PassengerId', 'Pclass', 'SibSp', 'Parch']]
y_train = train['Survived']
#print('Test shape:', test.shape)
#print('Test columns:', test.columns.tolist())
sample_submission = pd.read_csv('gender_submission.csv')
print(sample_submission.head())

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
test['Survived'] = logreg.predict(X_test)

submission = test[['PassengerId', 'Survived']]
submission.to_csv('first_sub.csv', index=False)
