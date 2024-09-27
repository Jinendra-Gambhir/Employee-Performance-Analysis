# -*- coding: utf-8 -*-
# !pip install pandas
# !pip install seaborn
# !pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/HR-Employee-Attrition.csv')
df.head()

df.shape

df.describe()

df.isnull().sum()

attrition_count = pd.DataFrame(df['Attrition'].value_counts())
attrition_count

plt.pie(attrition_count['count'], labels = ['No', 'Yes'], explode = (0.2, 0))

sns.countplot(data = df, x = 'Attrition')

df.drop(['EmployeeCount', 'EmployeeNumber'], axis = 1, inplace = True)

attrition_dummies = pd.get_dummies(df['Attrition'])
attrition_dummies.head()

df = pd.concat([df, attrition_dummies], axis=1)
df.head()

df.drop(['Attrition', 'No'], axis=1, inplace=True)

sns.barplot(data=df, x='Gender', y='Yes')

sns.barplot(data=df, x='Department', y='Yes')

sns.barplot(data=df, x='BusinessTravel', y='Yes')

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr()) # Selecting only numeric columns for correlation analysis

df.drop(['Age', 'JobLevel'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    else:
        df[column] = LabelEncoder().fit_transform(df[column])

df.head()

df.dtypes

"""**Model Building**"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

x = df.drop(['Yes'], axis=1)
y = df['Yes']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape

rf.fit(X_train, y_train)

rf.score(X_train, y_train)

"""**Prediction for test data**"""

pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(y_test, pred)

confusion_matrix(y_test, pred)
