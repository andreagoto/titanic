import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Train shape:', train.shape)
print('Train columns:', train.columns.tolist())
print(train.Survived.value_counts())
print(train.Pclass.value_counts())
print(train.Name.value_counts())
print(train.Sex.value_counts())
print(train.Age.value_counts())
print(train.SibSp.value_counts())
print(train.Parch.value_counts())
print(train.Ticket.value_counts())
print(train.Fare.value_counts())
print(train.Cabin.value_counts())
print(train.Embarked.value_counts())
print(train.describe())
print('Test shape:', test.shape)
print('Test columns:', test.columns.tolist())
