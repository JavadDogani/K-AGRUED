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
processrate=500.0
def normalarray(x):
    min1=min(x)
    y=[0,0,0,0,0]
    for i in range(len(x)): 
        val1=x[i]/min1
        val2=val1-1
        y[i]=round(val2/val1,2)
        x[i]=round(val2,2)
    return x,y

def plotdraw(data):
    style=['b-', 'r-', 'g-','y-']
    t = np.arange(0.0, data[0].shape[0], 1)    
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot(111)
    for i in range(len(data)):
         ax.plot(t,data[i],style[i%len(style)]) 
    fig = plt.figure(figsize=(25, 6))
    ax.set_xlabel('xlabel', fontsize=60)
    ax.set_ylabel('Workload (HTTP requst)', fontsize=40)
    ax.yaxis.set_tick_params(labelsize=35)
    ax.xaxis.set_tick_params(labelsize=35)
    plt.show()

def drawnormalchart(data,improve,metric,lbl1):
    n = [1,2,3,4,5]
    t = np.linspace(0,4,num=5)
    xy1=[0,1,2,3,4]
    fig1, ax = plt.subplots(1,figsize=(12,5))
    #a[0]=0.74
    norm=[1,1,1,1,1]
    plt.title('Google Cluster Dataset',fontsize=15)
    plt.ylabel('Normalized '+metric, fontsize=15)
    plt.xlabel('Methods', fontsize=15)
    plt.plot(t,norm,'-b')
    plt.bar(t,norm)
    plt.bar(t,data,bottom=norm, hatch = '.',label = 'df.A')
    plt.xticks(range(5), lbl1, size='small')
    plt.ylim(0,3)
    for i in range(3,-1,-1):
        plt.annotate("+"+str(improve[i]), xy=(xy1[i],data[i]+1.02), ha='center', va='bottom', fontsize=20)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13)
    plt.savefig(metric+'.svg',bbox_inches = 'tight')
    plt.show()  

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
    
lbl=['LSTM','SVR' , 'ARIMA', 'LSTM', 'GRUED']

for j in lbl[0:1]:
    Provisioning = np.array([])
    inputfilepath_predict="t+3.csv"
    
    
    real=pd.read_csv(inputfilepath_predict)['Real'].values
    podsarrive=np.ceil(real/processrate)#rt
    
    print("\n------------NO AutoScaling---------")
    pods_list=np.zeros(len(real))
    pods_list.fill(13)
    
    tetau_n=tetaU(pods_list, podsarrive)
    tetao_n=tetaO(pods_list, podsarrive)
    tu_n=TU(pods_list, podsarrive)
    to_n=TO(pods_list, podsarrive)
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
    
    epsilon=(tetau_n*tetao_n*tu_n*to_n)/(tetau_p*tetao_p*tu_p*to_p)
    epsilon=math.sqrt(math.sqrt(epsilon))
    print("epsilon%:",epsilon)
    print("count up down:",upcount,downcount)
    
    
    start=100
    end=start+300
    x=[pods_list[start:end],HPApods_list[start:end],podsarrive[start:end]]
    #x=[pods_list[:50],real[:50],podsarrive[:50],HPApods_list[:50]]
    plotdraw(x)
    #plotdraw(pods_list[:300],real[:300])
    
 