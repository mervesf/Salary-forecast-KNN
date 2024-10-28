# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:19:47 2023

@author: user
"""
#there is a correlation between error and salary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df=pd.read_csv('hitters.csv')
df['League']=[0 if i=='A' else 1 for i in df['League']]
df['Division']=[1 if i=='E' else 0 for i in df['Division']]
df['NewLeague']=[0 if i=='A' else 1 for i in df['NewLeague']]


x_test_1=df[pd.isna(df['Salary'])]
y_test_1=x_test_1[['Salary']]
x_test_1.drop('Salary',axis=1,inplace=True)
df.drop(x_test_1.index,axis=0,inplace=True)
df=df.reset_index(drop=True)
x=df.iloc[:,df.columns.values!='Salary']
y=df[['Salary']]




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
x_train=x_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

#Feature Analysis

pd.plotting.scatter_matrix(df,
                             c='red',
                            figsize= [15,15],
                             diagonal='hist',
                             alpha=0.5,
                             s = 200,
                             marker = '*',
                             edgecolor= "black")
plt.show()

corelation=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corelation,annot=True)
plt.show()



new_data=pd.DataFrame(index=df.index)
hit_pro=[df.iloc[i,1]/df.iloc[i,0] for i in df.index]
new_data['hit_pro']=hit_pro




def plot_player_stroke(data1):
    value=[(100*j)/(i) for i,j in data1[['AtBat','Runs']].values]
    data=[go.Bar(x=data1.index,
                 y=data1['AtBat'],
                 name="Players' total number of shots in a season",
                 marker=dict(color='rgb(123,22,78)')
        
        ),go.Bar(x=data1.index,
                     y=value,
                     name="Players' percentage of points scored for their team in total number of shots in a season",
                     marker=dict(color='rgb(219,78,128)'),
                     
            )]
    layout=go.Layout(barmode='overlay')
    figure=go.Figure(data=data,layout=layout)
    plot(figure)
    



best_contribution_player=sorted([(100*j)/(i) for i,j in df[['AtBat','Runs']].values],reverse=True)
major_years=[df['Years'].corr(df[i]) for i in df.columns]
catbat_years=[j/i for i,j in df[['Years','CAtBat']].values]
plt.scatter(catbat_years,df['Salary'],color='green')
plt.show()


# Hit rates of players going to better or worse leagues--> A major league
n_league=df[(df['NewLeague']==0)&(df['League']==1)]['Hits'].mean()
a_league=df[(df['NewLeague']==1)&(df['League']==0)]['Hits'].mean()



#Model Selection 
#KNN
def find_indices(array,k):
    new_data=pd.DataFrame(array).sort_values(by=0,ascending=True)
    return new_data.iloc[:k,:].index

def find_target(indices,y_labels,k):
    result=np.array(y_labels.iloc[indices].values).reshape(1,-1)
    result_1=[1.7-(i*0.2) for i in range(0,k)]
    print(result_1)
    return sum(result*result_1).mean()

def build_knn_model(x_train,y_train):
    x_samples=x_train
    y_labels=y_train
    return x_samples,y_labels


def predict_knn(x_samples,y_labels,x_test,k):
    distance_list=euclidean(x_samples,x_test)
    y_head=[]
    for i in distance_list:
        indices=find_indices(i,k)
        y_head.append(find_target(indices,y_labels,k))
        
        
    return y_head
        
def euclidean(x_data,x_data2):
        distance=[np.sum(np.power(i.reshape(1,-1)-np.array(x_data),2),axis=1) for i in np.array(x_data2)]
        return distance

def knn_classification(k,x_train,y_train,x_test,y_test):
    x_samples,y_labels=build_knn_model(x_train,y_train)
    y_head=predict_knn(x_samples,y_labels,x_test,k)
    return y_head
    
y_head=knn_classification(5,x_train,y_train,x_test,y_test)
mse = mean_squared_error(y_test, y_head)
r2 = r2_score(y_test, y_head)
       





