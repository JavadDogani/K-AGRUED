from __future__ import print_function
import pandas as pd
import numpy as np
#122067
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import random
import math
processrate=10.0
def normalarray(x):
    min1=min(x)
    y=[0,0,0,0,0]
    for i in range(len(x)): 
        val1=x[i]/min1
        val2=val1-1
        y[i]=round(val2/val1,2)
        x[i]=round(val2,2)
    return x,y



def CheckProvisioning(data):
    upcount=0
    downcount=0
    Len=len(data)
    #print("len:",Len)
    pods_list=np.empty([Len])
    pods_list.fill(-1.0)
    #print(pods_list)
    gdt=0.6
    i=0
    pods=0
    j=0
    while(j<Len):
        #print("\n=========time",j," pods is:", pods, "data:",data[j])
        pods_need=math.ceil(max(data[j:j+3])/processrate)
        #print("need:",pods_need)
        if (pods_need>pods):
            #maxfuture=max(data[j:j+3])
            #pods_need=math.ceil(data[j]/processrate)
        
            
            #scale up
            upcount+=1
            pods=pods_need
            pods_list[j]=pods
            j+=3
        elif (math.ceil(max(data[j:j+5])/processrate)<pods):
            pods_need=math.ceil(max(data[j:j+5])/processrate)
            diff=pods-pods_need
            pods_remove=math.ceil(gdt*diff)
            #scaledown
            downcount+=1
            pods-=pods_remove
            #print("scale down in:",j," to-",pods,"- by remove", pods_remove, "and diff:",diff)
           
            pods_list[j]=pods
            j+=5
        else:
            pods_list[j]=pods
            #print("no change in:",j)
            j+=1
    return pods_list,upcount,downcount

def HPACheckProvisioning(data):
    upcount=0
    downcount=0
    
    Len=len(data)
    pods_list=np.empty([Len])
    pods_list.fill(-1.0)
    gdt=0.6
    i=0
    pods=0
    j=0
    while(j<Len):
        #print("\n=========time",j," pods is:", pods, "data:",data[j])
        pods_need=math.ceil(data[j]/processrate)
        #print("need:",pods_need)
        if (pods_need>pods):
            #maxfuture=max(data[j:j+3])
            #pods_need=math.ceil(data[j]/processrate)
        
            
            #scale up
            upcount+=1
            pods=pods_need
            pods_list[j]=pods
            j+=3
        elif (math.ceil(data[j]/processrate)<pods):
            pods_need=math.ceil(data[j]/processrate)
            diff=pods-pods_need
            pods_remove=math.ceil(gdt*diff)
            #scaledown
            pods-=pods_remove
            downcount+=1
            
            pods_list[j]=pods
            j+=5
        else:
            pods_list[j]=pods
            #print("no change in:",j)
            j+=1
    return pods_list,upcount,downcount

def PreviousstudyCheckProvisioning(data):
    upcount=0
    downcount=0
    
    Len=len(data)
    pods_list=np.empty([Len])
    pods_list.fill(-1.0)
    gdt=0.6
    i=0
    pods=0
    j=0
    while(j<Len):
        #print("\n=========time",j," pods is:", pods, "data:",data[j])
        pods_need=math.ceil(data[j]/processrate)
        #print("need:",pods_need)
        if (pods_need>pods):
            #maxfuture=max(data[j:j+3])
            #pods_need=math.ceil(data[j]/processrate)
        
            
            #scale up
            upcount+=1
            pods=pods_need
            pods_list[j]=pods
            j+=1
        elif (math.ceil(data[j]/processrate)<pods):
            pods_need=math.ceil(data[j]/processrate)
            diff=pods-pods_need
            pods_remove=math.ceil(gdt*diff)
            #scaledown
            pods-=pods_remove
            downcount+=1
            pods_list[j]=pods
            j+=1
        else:
            pods_list[j]=pods
            #print("no change in:",j)
            j+=1
    return pods_list,upcount,downcount


def fillPodslist(pods_list):
    value=pods_list[0]
    Len=len(pods_list)
    #print(Len)
    for i in range (1,Len):
        if (pods_list[i]==-1):
            pods_list[i]=value
        else:
            value=pods_list[i]
def sgn(a,b):
    if a>b:
        return 1
    elif a<b:
        return -1
    else:
        return 0
def tetaU(pt,rt):
    #print("pt:",pt)
    #print("rt:",rt)
    sum=0.0
    for i in range (len(rt)):
        #print("i:",i)
        delta=0
        j=i
        if(rt[i]!=0):
            while(rt[j]==rt[i] and j>=0):
                j=j-1
                delta+=1        
            val=(max(rt[i]-pt[i],0)*delta)/rt[i]
            
            sum+=val
        #print("delta:", delta, "val:", val,"sum:", sum)
    return sum/len(rt)*100

    
def tetaO(pt,rt):
    sum=0.0
    for i in range (len(rt)):
        #print("i:",i)
        delta=0
        j=i
        if (pt[i]!=0):
            while(rt[j]==rt[i] and j>=0):
                j=j-1
                delta+=1        
            val=(max(pt[i]-rt[i],0)*delta)/pt[i]
        
        sum+=val
        
    return sum/len(rt)*100

def TU(pt,rt):
    #print("pt:",pt)
    #print("rt:",rt)
    sum=0.0
    for i in range (len(rt)):
        #print("i:",i)
        delta=0
        j=i
        while(rt[j]==rt[i] and j>=0):
            j=j-1
            delta+=1        
        val=max(sgn(rt[i],pt[i]),0)*delta
        
        sum+=val
        #print("delta:", delta, "val:", val,"sum:", sum)
    return sum/len(rt)*100

    
def TO(pt,rt):
    sum=0.0
    for i in range (len(rt)):
        #print("i:",i)
        delta=0
        j=i
        while(rt[j]==rt[i] and j>=0):
            j=j-1
            delta+=1        
        val=max(sgn(pt[i],rt[i]),0)*delta       
        sum+=val
        
    return sum/len(rt)*100
    

lbl=['GRUED','SVR' , 'ARIMA', 'LSTM', 'GRUED']

for j in lbl[0:1]:
    Provisioning = np.array([])
    inputfilepath_predict="t+3.csv"
    
    
    real=pd.read_csv(inputfilepath_predict)['Real'].values
    podsarrive=np.ceil(real/processrate)#rt
    
    print("\n------------NO AutoScaling---------")
    pods_list_n=np.zeros(len(real))
    pods_list_n.fill(4)
    
    tetau_n=tetaU(pods_list_n, podsarrive)
    tetao_n=tetaO(pods_list_n, podsarrive)
    tu_n=TU(pods_list_n, podsarrive)
    to_n=TO(pods_list_n, podsarrive)
    print("TetaU%:",tetau_n)
    print("TetaO%:",tetao_n)
    print("TU%:",tu_n)
    print("TO%:",to_n)
    print("epsilon%:1")
    
    data=pd.read_csv(inputfilepath_predict)[j].values
    print("\n------------OURS---------")
    
    pods_list,upcount,downcount=CheckProvisioning(data)
    fillPodslist(pods_list)#pt
    tetau_m=tetaU(pods_list, podsarrive)
    tetao_m=tetaO(pods_list, podsarrive)
    tu_m=TU(pods_list, podsarrive)
    to_m=TO(pods_list, podsarrive)
    print("TetaU%:",tetau_m)
    print("TetaO%:",tetao_m)
    print("TU%:",tu_m)
    print("TO%:",to_m)
    epsilon=(tetau_n*tetao_n*tu_n*to_n)/(tetau_m*tetao_m*tu_m*to_m)
    epsilon=math.sqrt(math.sqrt(epsilon))
    print("epsilon%:",epsilon)
    print("count up down:",upcount,downcount)
    
    print("\n------------HPA---------")
    HPApods_list,upcount,downcount=HPACheckProvisioning(real)
    fillPodslist(HPApods_list)#pt
    tetau_h=tetaU(HPApods_list, podsarrive)
    tetao_h=tetaO(HPApods_list, podsarrive)
    tu_h=TU(HPApods_list, podsarrive)
    to_h=TO(HPApods_list, podsarrive)
    print("TetaU%:",tetau_h)
    print("TetaO%:",tetao_h)
    print("TU%:",tu_h)
    print("TO%:",to_h)
    epsilon=(tetau_n*tetao_n*tu_n*to_n)/(tetau_h*tetao_h*tu_h*to_h)
    epsilon=math.sqrt(math.sqrt(epsilon))
    print("epsilon%:",epsilon)
    print("count up down:",upcount,downcount)
    
    print("count up down:",upcount,downcount)
    
    print("\n------------Previous study---------")
    Prevpods_list,upcount,downcount=PreviousstudyCheckProvisioning(data)
    fillPodslist(Prevpods_list)#pt
    tetau_p=tetaU(Prevpods_list, podsarrive)
    tetao_p=tetaO(Prevpods_list, podsarrive)
    tu_p=TU(Prevpods_list, podsarrive)
    to_p=TO(Prevpods_list, podsarrive)
    print("TetaU%:",tetau_p)
    print("TetaO%:",tetao_p)
    print("TU%:",tu_p)
    print("TO%:",to_p)
    
    epsilon=(tetau_n*tetao_n*tu_n*to_n)/(tetau_h*tetao_h*tu_h*to_h)
    epsilon=math.sqrt(math.sqrt(epsilon))
    print("epsilon%:",epsilon)
    print("count up down:",upcount,downcount)
    
    
    start=1000
    step=500
    x=[pods_list[start:start+step],HPApods_list[start:start+step],podsarrive[start:start+step]]

    
 