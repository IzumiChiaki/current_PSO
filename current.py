# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:17:09 2019

@author: Chiaki
"""

import sys
sys.path.append("D:\Python\coating")

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est='mic_approx')

# import original data
df = pd.read_csv('./QDalldatachuli.csv')
# delete unused columns
df1 = df.drop(['date','volume','CO','NO2','O3','PM10','SO2','num'], axis=1)
df1_4 = df.drop(['date','volume','CO','NO2','AQI','PM10','SO2','num'], axis=1)
# get header lists
cols = list(df1)
cols_4 = list(df1_4) 
# move 'current' to N0.4 column to get new header lists
cols.insert(4, cols.pop(cols.index('current'))) 
cols_4.insert(4, cols_4.pop(cols_4.index('current')))
# use loc() and get new dataframe for mic analysis
df2 = df1.loc[:,cols] 
df2_4 = df1_4.loc[:,cols_4] 
# use df2 minus df2.mean() to get new dataframe for pearson analysis
df3 = df2 - df2.mean()
df4 = df2.drop(['AQI'],axis=1)
df5 = df4 - df4.mean()
# get current_PSO.csv for PSO using mic analysis
df4.to_csv('current_PSO_mic_3.csv',index = False)
df2.to_csv('current_PSO_mic_4.csv',index = False) 
df2_4.to_csv('current_PSO_mic_4_test.csv',index = False) 
# get THAQIPM25I.txt for PSO using pearson analysis
df3.to_csv('THAQIPM25I4.txt',index = False, header = False) 
df5.to_csv('THAQIPM25I3.txt',index = False, header = False) 
# import the result of mic by PSO
result_mic_4 = pd.read_csv('./result_PSO_mic_4.csv')
result_mic_3 = pd.read_csv('./result_PSO_mic_3.csv')
# import the result of pearson by PSO
result_p_4 = pd.read_csv('./result_PSO_pearson_4.csv')
result_p_3 = pd.read_csv('./result_PSO_pearson_3.csv')  
# get the max of MIC
MIC_max_4 = result_mic_4['MIC'].max() 
MIC_max_3 = result_mic_3['MIC'].max() 
# get the max of pearson
P_max_4 = result_p_4['Pearson'].max() 
P_max_3 = result_p_3['Pearson'].max() 
# get the index of MIC_max
Index_mic_max_4 = result_mic_4['MIC'].argmax() 
Index_mic_max_3 = result_mic_3['MIC'].argmax() 
# get the index of pearson_max
Index_p_max_4 = result_p_4['Pearson'].argmax() 
Index_p_max_3 = result_p_3['Pearson'].argmax() 
print('==========Max MIC 4==========')
print(MIC_max_4)
print('T       H       AQI     PM2.5')
print(result_mic_4.values[Index_mic_max_4,0:4])
print('==========Max MIC 3==========')
print(MIC_max_3)
print('T       H       PM2.5')
print(result_mic_3.values[Index_mic_max_3,0:3])
print('========Max Pearson 4========')
print(P_max_4)
print('T       H       AQI     PM2.5')
print(result_p_4.values[Index_p_max_4,0:4])
print('========Max Pearson 3========')
print(P_max_3)
print('T       H       PM2.5')
print(result_p_3.values[Index_p_max_3,0:3])

df_r = df.drop(['date','num'], axis=1)
pearson_list = []
mic_list = []
mine = MINE(alpha=0.6, c=15, est='mic_approx')
for i in range(df_r.values.shape[1]-1):
    a = np.array(pearsonr(df_r.values[:,i+1], df_r.values[:,0]))
    pearson_list.append(a)
    b = mine.compute_score(df_r.values[:,i+1], df_r.values[:,0])
    mic_list.append(mine.mic())
pearson = np.array(pearson_list)
mic = np.array(mic_list)
print("==================pearson_r===================")
print(pearson[:,0])
print("=====================mic======================")
print(mic)

dfData = df_r.corr()
plt.subplots(figsize=(9, 9)) #set figsize
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()

factor = [0.546,0.013,0.066]
data_p =np.array([[df4.values[i][j]**2 for j in range(len(df4.values[i])-1)] for i in range(len(df4.values))])
E_list = np.sqrt(data_p[:,0:3].astype(np.float64).dot(np.array(factor).T))
df4['E'] = E_list
colsE = list(df4)
colsE.insert(3, colsE.pop(colsE.index('E'))) 
# use loc() and get new dataframe for mic analysis
df4E = df4.loc[:,colsE] 
df4E.to_csv('E_plot.csv',index = False)