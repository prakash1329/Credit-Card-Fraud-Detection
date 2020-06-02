#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from imblearn.over_sampling import SMOTE


from warnings import filterwarnings
filterwarnings('ignore')


# # Data Exploration & Feature Engineering

# In[2]:


data = pd.read_csv('creditcard.csv')
print(f'Credit Card Fraud Detection Data Size - {data.shape}')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


amount_descp =  data['Amount'].describe()
time_descp =  data['Time'].describe()

print('Amount Statistics:\n\n',amount_descp)
print('\nTime Statistics:\n\n',time_descp)


# In[8]:


data.isna().any()


# # Exploratory Data Analysis

# In[9]:


plt.figure(figsize=(7,7))
sns.countplot(data.Class, palette = 'viridis')
plt.title('Class Distributions \n 0: Non-Fraud and 1: Fraud', y= 1.01, fontsize = 20)
plt.ylabel('Number of Records')
plt.show()


# In[10]:


data.corrwith(data.Class).plot(kind = 'bar',figsize = (20, 10),fontsize = 15,rot = 90,
                               title = "Correlation with class", grid = True)
plt.show()


# In[11]:


plt.subplots(figsize = (25,10))
sns.heatmap(data.corr(), square = True, cmap="YlGnBu", lw = 0.5)


# In[12]:


fig, ax = plt.subplots(1,2, figsize = (25,7))

ax[0].hist(data['Amount'], bins = 80, color = 'orange')
ax[0].set_title('Transaction Amount Distribution', fontsize = 15)

ax[1].hist(data['Time'], bins = 60, color = 'green')
ax[1].set_title(' Transaction Time Distribution',fontsize = 15)
plt.show()


# In[13]:


non_fraud_data = data[data['Class'] == 0]
fraud_data = data[data['Class'] == 1]


predictors = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13',
              'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
              'V25', 'V26', 'V27', 'V28','Amount']

fig, ax = plt.subplots(1, 2, figsize = (25,7))
fig.suptitle('Amount transaction distributed by class', fontsize=25)

ax[0].hist(fraud_data.Amount, bins = 80, color ='violet')
ax[0].set_title('Fraud', fontsize = 15)

ax[1].hist(non_fraud_data.Amount, bins = 60, color = 'yellow')
ax[1].set_title('Non - Fraud', fontsize = 15)
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[14]:


fig, ax = plt.subplots(1, 2, figsize=(25, 7))
fig.suptitle('Time taken vs Amount transaction distributed by class', fontsize=25)

ax[0].scatter(fraud_data.Time, fraud_data.Amount, color='red')
ax[0].set_title('Fraud', fontsize = 15)

ax[1].scatter(non_fraud_data.Time, non_fraud_data.Amount)
ax[1].set_title('Non-Fraud' , fontsize = 15)

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[15]:


data['Normalized_Amount'] = RobustScaler().fit_transform(data['Amount'].values.reshape(-1,1))
fields_to_drop = ['Amount','Time']
data = data.drop(fields_to_drop,axis=1)
data.head()


# In[16]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
y.head()


# # Train - Test Split

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=13)

X_train,X_test = X_train.values, X_test.values
y_train,y_test = y_train.values, y_test.values

print(f' X Train : {X_train.shape}')
print(f' X Test  : {X_test.shape}')
print(f' y Train : {y_train.shape}')
print(f' y Test  : {y_test.shape}')


# # Fraud Detection Modeling

# In[18]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ## Logistic Regression

# In[19]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
print('Default Parameters Logistic Regression:\n')
print(f'Logistic Regression Parameters:\n {log_reg}')
print(f' Default Score: {log_reg.score(X_test,y_test)}\n')

print('Hyper-Tuning Logistic Regression:\n')

log_reg_params = {"penalty": ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

log_reg_grid = GridSearchCV(LogisticRegression(), log_reg_params)
log_reg_grid.fit(X_train, y_train)
log_reg = log_reg_grid.best_estimator_
print(f' Tuned Parameters:\n {log_reg_grid.best_estimator_}')
print(f' Tuned Params: {log_reg_grid.best_params_}\n')
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

log_reg_score = log_reg.score(X_test, y_test)
print(f' Hyper-Tuned Score: {log_reg_score}\n')

log_reg_cnf_matrix = confusion_matrix(y_test, log_reg_preds)
plot_confusion_matrix(log_reg_cnf_matrix, classes=[0, 1])
plt.show()


# ## Decision-Tree Classifier

# In[20]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
decision_tree_preds = decision_tree.predict(X_test)
print('Default Parameters Decision Tree:\n')
print(f'Decision Tree Parameters:\n {decision_tree}')
print(f' Default Score: {decision_tree.score(X_test,y_test)}\n')

print('Hyper-Tuning Decision Tree:\n')

decision_tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 10, 1)),
                        "min_samples_leaf": list(range(2, 10, 1))}

decision_tree_grid = GridSearchCV(
    DecisionTreeClassifier(), decision_tree_params)
decision_tree_grid.fit(X_train, y_train)
decision_tree = decision_tree_grid.best_estimator_
print(f' Tuned Parameters:\n {decision_tree_grid.best_estimator_}')
print(f' Tuned Params: {decision_tree_grid.best_params_}\n')
decision_tree.fit(X_train, y_train)
decision_tree_preds = decision_tree.predict(X_test)

decision_tree_score = decision_tree.score(X_test,y_test)
print(f' Hyper-Tuned Score: {decision_tree_score}\n')

decision_tree_cnf_matrix = confusion_matrix(y_test, decision_tree_preds)
plot_confusion_matrix(decision_tree_cnf_matrix, classes=[0, 1])
plt.show()


# ## Random Forest Classifier

# In[21]:


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
random_forest_preds = random_forest.predict(X_test)
print('Default Random Forest:\n')
print(f' Parameters:\n {random_forest}')
print(f' Default Score: {random_forest.score(X_test,y_test)}\n')

print('Hyper-Tuning Random Forest:\n')

random_forest_params = {"n_estimators": [200, 250, 300],
                        "bootstrap": [True, False],
                        "max_depth": [15, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [3,5]}

random_forest_grid = GridSearchCV(
    RandomForestClassifier(), random_forest_params)
random_forest_grid.fit(X_train, y_train)
random_forest = random_forest_grid.best_estimator_
print(f' Tuned Parameters:\n {random_forest_grid.best_estimator_}')
print(f' Tuned Params: {random_forest_grid.best_params_}\n')
random_forest.fit(X_train, y_train)
random_forest_preds = random_forest.predict(X_test)

random_forest_score = random_forest.score(X_test, y_test)
print(f' Hyper-Tuned Score: {random_forest_score}\n')

random_forest_cnf_matrix = confusion_matrix(y_test, random_forest_preds)
plot_confusion_matrix(random_forest_cnf_matrix, classes=[0, 1])
plt.show()

rf_feature_imp = pd.DataFrame({'Feature': predictors, 'Feature importance': random_forest.feature_importances_})
rf_feature_imp = rf_feature_imp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
s = sns.barplot(x='Feature',y='Feature importance',data= rf_feature_imp)
plt.title('Random Forest - Features importance',fontsize=15, y=1.01)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


# ## XGBoost

# In[23]:


dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

watchlist = [(dtrain, 'train')]

xgb_params = {'booster': 'gbtree',
              'tree_method': 'gpu_hist',
              'objective': 'binary:logistic',
              'eta': 0.020,
              'silent': True,
              'max_depth': 15,
              'subsample': 0.8,
              'colsample_bytree': 0.9,
              'colsample_bylevel': 0.5,
              'eval_metric': 'auc',
              'random_state': 13
              }

xgb_model = xgb.train(xgb_params, dtrain, 501, watchlist, early_stopping_rounds=50,
                      maximize=True, verbose_eval=100)
xgb_preds = xgb_model.predict(dtest)

fig, ax = plt.subplots(figsize=(10, 7))
xgb.plot_importance(xgb_model, height=0.8, title=" XGBoost : Features importance", ax=ax,
                    color="red")
plt.show()


# # Deep Neural Network - ANN

# In[25]:


ANN_model = Sequential([
    Dense(units=8, input_dim = 29,activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])

ANN_model.summary()
print('')
print('Training Model:\n')
ANN_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ANN_model.fit(X_train,y_train,batch_size=16,epochs=100)
ANN_pred = ANN_model.predict(X_test)

print('')
print('Testing Model: \n')
ANN_acc_score = ANN_model.evaluate(X_test, y_test)
print('')
print(f'ANN Model Test Score: {ANN_acc_score}\n')
      
ANN_cnf_matrix = confusion_matrix(y_test,ANN_pred.round())
plot_confusion_matrix(ANN_cnf_matrix, classes=[0,1])
plt.show()


# ## Under Sampling

# In[26]:


fraud_indices = np.array(data[data.Class == 1].index)
nonfraud_indices = data[data.Class == 0].index
number_records_fraud = len(fraud_indices)
print(f' Number of Fraud Transactions: {number_records_fraud}')

random_nonfraud_indices = np.random.choice(nonfraud_indices, number_records_fraud, replace=False)
random_nonfraud_indices = np.array(random_nonfraud_indices)
number_records_nonfraud = len(random_nonfraud_indices)
print(f' Number of Non-Fraud Transactions: {number_records_nonfraud}')

under_sample_indices = np.concatenate([fraud_indices,random_nonfraud_indices])
number_records_under_sampled = len(under_sample_indices)
print('')
print(f' Number of Under-Sampled Data Generated : {number_records_under_sampled}')

under_sample_data = data.iloc[under_sample_indices,:]
X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)

X_train,X_test = X_train.values, X_test.values
y_train,y_test = y_train.values, y_test.values

ANN_model.summary()
print('')
print('Training Model:\n')
ANN_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ANN_model.fit(X_train,y_train,batch_size=16,epochs=100)
under_sample_ANN_pred = ANN_model.predict(X_test)

print('')
print('Testing Model: \n')
under_sample_ANN_acc_score = ANN_model.evaluate(X_test, y_test)
print('')
print(f'ANN Model Test Score: {under_sample_ANN_acc_score}\n')
      
under_sample_ANN_cnf_matrix = confusion_matrix(y_test,under_sample_ANN_pred.round())
plot_confusion_matrix(under_sample_ANN_cnf_matrix, classes=[0,1])
plt.show()


# ## SMOTE Sampling

# In[27]:


X_resample, y_resample = SMOTE().fit_sample(X,y)
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)

X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

X_train,X_test = X_train.values, X_test.values
y_train,y_test = y_train.values, y_test.values

ANN_model.summary()
print('')
print('Training Model:\n')
ANN_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ANN_model.fit(X_train,y_train,batch_size=16,epochs=100)
smote_ANN_pred = ANN_model.predict(X_test)

print('')
print('Testing Model: \n')
smote_ANN_acc_score = ANN_model.evaluate(X_test, y_test)
print('')
print(f'SMOTE ANN Model Test Score: {smote_ANN_acc_score}\n')
      
smote_ANN_cnf_matrix = confusion_matrix(y_test,smote_ANN_pred.round())
plot_confusion_matrix(under_sample_ANN_cnf_matrix, classes=[0,1])
plt.show()


# In[ ]:




