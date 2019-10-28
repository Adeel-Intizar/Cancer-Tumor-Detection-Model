#!/usr/bin/env python
# coding: utf-8

# Importing Essential Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump


# Loading Data

data = load_breast_cancer()
data.keys()

data.target_names


data.feature_names


df1 = pd.DataFrame(data['data'], columns=data.feature_names)
df2 = pd.Series(data.target, name='Class')
df2.value_counts()


df = pd.concat([df1, df2], axis=1)
df.head()


z = df1.copy().values
pca = PCA(n_components=2)
pc = pca.fit_transform(z)
pcdf = pd.DataFrame(pc, columns=['1', '2'])
pcdf = pd.concat([pcdf, df2], axis = 1)
pcdf.head()


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC 1', fontsize=13)
ax.set_ylabel('PC 2', fontsize=13)
ax.set_title('PCA Visualization', fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    index = pcdf['Class'] == target
    ax.scatter(pcdf.loc[index, '1'], pcdf.loc[index, '2'], c = color, s = 50)
ax.legend(['Malignant', 'Benign'])
ax.grid()



df.hist(figsize=(15,15))
plt.show()


# Finding Correlation

cor = df.corr()
corr = abs(cor['Class'])
imp = corr[corr > 0.5]
imp


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target)


# Model Performaces

models = [LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced', random_state=42),
         RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 
         SVC(kernel='linear', gamma='auto', class_weight='balanced'),
         GradientBoostingClassifier(), DecisionTreeClassifier(class_weight='balanced', random_state=42)]

classifiers = ['Logistic Reg', 'Random Forest', 'Linear_SVC', 'Gradient Boosting', 'Decision Tree']

for (model, name) in zip(models, classifiers):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = (y_pred != y_test).sum()
    cv = cross_val_score(model, df1, df2, cv=5, scoring='accuracy')
    print('{}: '.format(name))
    print('CV Mean: {}'.format(np.mean(cv)))
    print('Errors: {}'.format(errors))
    print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))
    print('ROC_AUC_Score: {}'.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Matrix: \n {}'.format(confusion_matrix(y_test, y_pred)))
    print('Classification Report: \n {} \n\n'.format(classification_report(y_test, y_pred)))


# Model Performances with Scaling and Dimensionality Reduction

models = [LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced', random_state=42),
         RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 
         SVC(kernel='linear', gamma='auto', class_weight='balanced'),
         GradientBoostingClassifier(), DecisionTreeClassifier(class_weight='balanced', random_state=42)]

classifiers = ['Logistic Reg', 'Random Forest', 'Linear_SVC', 'Gradient Boosting', 'Decision Tree']

for (model, name) in zip(models, classifiers):
    model = model
    scaler = StandardScaler()
    pca = PCA(.95)
    pipeline = make_pipeline(scaler, pca, model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    errors = (y_pred != y_test).sum()
    cv = cross_val_score(pipeline, df1, df2, cv=5, scoring='accuracy')
    print('{}: '.format(name))
    print('CV Mean: {}'.format(np.mean(cv)))
    print('Errors: {}'.format(errors))
    print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))
    print('ROC_AUC_Score: {}'.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Matrix: \n {}'.format(confusion_matrix(y_test, y_pred)))
    print('Classification Report: \n {} \n\n'.format(classification_report(y_test, y_pred)))


# Choosing Linear SVC

scaler = StandardScaler()
pca = PCA(.95)
model = SVC(kernel='linear', gamma='auto', class_weight='balanced')
pipeline = make_pipeline(scaler, pca, model)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Accuracy:',accuracy_score(y_test, y_pred))
print('Errors:', (y_pred != y_test).sum())


# Finalizing and Saving the Model

final = make_pipeline(scaler, pca, model)
final.fit(df1, df2)
dump(final, 'Cancer Tumor Detection Model.pkl')
