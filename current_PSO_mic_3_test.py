# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:17:29 2019

@author: Chiaki
"""
import sys
sys.path.append("D:\Python\current")

import numpy as np
import pandas as pd
import copy as npcopy
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est='mic_approx')

import time

class MyPSO(object):
    """
    pop_size：鸟群规模
    factor_size：解的维度
    wmax, wmin ：惯性权值 w 的取值范围
    c1, c2：学习参数
    iter：最大迭代次数
    """
    def __init__(self,pop_size,factor_size,wmax,wmin,c1,c2,iter,data):
        self.pop_size = pop_size
        self.factor_size = factor_size
        self.wmax = wmax
        self.wmin = wmin
        self.c1 = c1
        self.c2 = c2
        self.iter = iter
        self.data =  data

    def Initpart(self):
        initvect, initvectdict, initbestpdict, initbestg, initresultdict = [], {}, {}, [], {}
        initvect = np.random.random_sample(self.factor_size)
        initbestg = npcopy.deepcopy(initvect)
        for i in range(self.pop_size):
            initvectdict[i] = initvect
            initbestpdict[i] = npcopy.deepcopy(initvect)
            initresultdict[i] = float('0')
        return initvectdict,initbestpdict,initbestg,initresultdict

    def Func(self,factor):
        data_p =np.array([[self.data.values[i][j]**2 for j in range(len(self.data.values[i])-1)] for i in range(len(self.data.values))])
        E_list = np.sqrt(data_p[:,0:3].astype(np.float64).dot(np.array(factor).T))
        pr_list = []
        mine.compute_score(E_list, self.data.values[:,3])
        pr_list.append(mine.mic())
        pr = np.array(pr_list)
        correlation = pr[0]
        return correlation

    def Vector(self,iternow,vectnow,bestp,bestg):
        r1, r2 = np.random.random(), np.random.random()
        vectnext = {}
        wnow = self.wmax - (self.wmax - self.wmin) / self.iter * iternow
        for i in range(self.pop_size):
            vectnext[i] = wnow * vectnow[i] + self.c1*r1*(bestp[i] - vectnow[i]) + self.c2*r2*(bestg - vectnow[i])
        return vectnext

    def Run(self):
        resultmax = float('0')
        iterresult,iterresultdict = [],{}
        vectdict, bestp, bestg, resultdict = self.Initpart()
        for iter in range(self.iter):
            for j in range(self.pop_size):
                resultj = self.Func(vectdict[j])
                if resultj > resultdict[j]:
                    resultdict[j] = resultj
                    bestp[j] = vectdict[j]
                else:
                    pass
                iterresult.append(resultj)
            iterresultdict[iter] = max(iterresult)
            for index, result in resultdict.items():
                if result > resultmax:
                    resultmax = result
                    bestg = bestp[index]
                else:
                    pass
            if resultmax > 0:
                print("Best result = " + str(resultmax))
                vector = [round(a,3) for a in bestg.tolist()]
                print("Best vector = " + str(vector))
                #self.Ploterro(iterresultdict)
                break
            else:
                vectdict = self.Vector(iter,vectdict,bestp,bestg)
        return resultj, vector
    
if __name__ == "__main__":
    starttime = time.time()
    data = pd.read_csv('current_PSO_mic_3.csv')
    result = []
    for i in range(50):
        print("iter",i)
        a = MyPSO(60,3,0.9,0.4,2.05,2.05,500,data=data)
        result.append(a.Run())
    res = max(result)
    endtime = time.time()
    dataframe = pd.DataFrame({'MIC':[res[0]],'T factor':[res[1][0]],\
                              'H factor':[res[1][1]],'PM2.5 factor':[res[1][2]]})
    dataframe.to_csv("result_PSO_mic_3_test.csv",mode='a',index=False,header=False,sep=',',\
                     columns=['T factor','H factor','PM2.5 factor','MIC'])
    print(res)
    print("Runtime = " + str(endtime - starttime))

