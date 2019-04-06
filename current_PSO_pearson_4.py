#coding:utf-8
"""
"""
import numpy as np
import pandas as pd
import math
import copy as npcopy#copy.copy 浅拷贝 只拷贝父对象，不会拷贝对象的内部的子对象。copy.deepcopy 深拷贝 拷贝对象及其子对象
import matplotlib as mpl
import matplotlib.pyplot as plt
#为了方便，不然要再写成mpl.pyplot麻烦
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
        samplenum = np.shape(self.data)[0]
        f1 = []; f2 = []; f3 = []; f11 = []; f22 = [];f33 = []
        for i in range(samplenum):
            f11 = (-factor[0] * self.data[i][0]+factor[1] * self.data[i][1] - \
                   factor[2] * self.data[i][2]-factor[3] * self.data[i][3]) * self.data[i][4]
                   
            f1.append(f11)
            f22=  (factor[0] * self.data[i][0]+factor[1] * self.data[i][1] + \
                              factor[2] * self.data[i][2]+factor[3] * self.data[i][3]) *(factor[0] * self.data[i][0] + \
                              factor[1] * self.data[i][1] + factor[2] * self.data[i][2]+factor[3] * self.data[i][3])
            f2.append(f22)
            f33=   self.data[i][4]*self.data[i][4]
            f3.append(f33)                  
        fenzi = sum(f1)
        fenmu1 = math.sqrt(sum(f2))
        fenmu2 = math.sqrt(sum(f3))
        correlation = abs(fenzi/fenmu1/fenmu2)
        return correlation

    def Vector(self,iternow,vectnow,bestp,bestg):
        r1, r2 = np.random.random(), np.random.random()
        vectnext = {}
        wnow = self.wmax - (self.wmax - self.wmin) / self.iter * iternow
        for i in range(self.pop_size):
            vectnext[i] = wnow * vectnow[i] + self.c1*r1*(bestp[i] - vectnow[i]) + self.c2*r2*(bestg - vectnow[i])
        return vectnext

    def Ploterro(self,errodict):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        plt.plot(errodict.keys(),errodict.values(),'r-',linewidth=1.5,markersize=5)
        ax.set_xlabel(u'迭代次数',fontsize=18)
        ax.set_ylabel(u'误差',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0,)
        plt.ylim(0,)
        plt.grid(True)
        plt.show()

    def Save(self,bestg,result):
        factorandrresult = bestg.ravel().tolist()
        factorandrresult.extend(result)
        file = open(r'PSO_DRF_CLASS2.txt','a')
        for i in range(np.shape(factorandrresult)[0]):
            if i == 4:
                file.write("   ")
                file.write(str(round(factorandrresult[i],4)))
                file.write(" ")
            else:
                file.write(str(round(factorandrresult[i],4)))
                file.write(" ")
        file.write("\n")
        file.close()

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
                vector = [round(a,4) for a in bestg.tolist()]
                print("Best vector = " + str(vector))
                #self.Ploterro(iterresultdict)
                break
            else:
                vectdict = self.Vector(iter,vectdict,bestp,bestg)
        return resultj, vector

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,5))
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:5]
        index += 1
    return returnMat
    
if __name__ == "__main__":
    starttime = time.time()
    data = file2matrix('THAQIPM25I4.txt')
    result = []
    for i in range(20):
        print("iter",i)
        a = MyPSO(60,4,0.9,0.4,2.05,2.05,100,data=data)
        result.append(a.Run())
    endtime = time.time()
    res = max(result)
    dataframe = pd.DataFrame({'Pearson':[res[0]],'T factor':[res[1][0]],\
                              'H factor':[res[1][1]],'AQI factor':[res[1][2]],\
                              'PM2.5 factor':[res[1][3]]})
    dataframe.to_csv("result_PSO_pearson_4.csv",mode='a',index=False,header=False,sep=',',\
                     columns=['T factor','H factor','AQI factor','PM2.5 factor','Pearson'])
    print(res)
    print("Runtime = " + str(endtime - starttime))

