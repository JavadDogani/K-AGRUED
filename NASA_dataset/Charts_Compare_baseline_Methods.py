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
lbl=['Real', 'LSTM', 'GRUED', 'K-AGRUED']
t = np.linspace(0,100,num=100)
xticks = np.arange(0, 101, 10)
plt.rcParams['axes.xmargin'] = 0.005
for m in range(3,6):
    inputfilepath_predict="t+"+str(m)+".csv"
    forecast=pd.read_csv(inputfilepath_predict)
    fig1, ax = plt.subplots(1,figsize=(50,20))
    symbol=['-b','-g','-k','-r']
    c=0
    for i in lbl: 
        j=np.where(forecast.columns.values==i)
        start=3567
        d1=forecast[forecast.columns[j]][start:start+100]
        plt.plot(t, d1.values,str(symbol[c]),label=i,markersize=6)
        c+=1    
    plt.title('NASA dataset (t+'+str(m)+')', fontsize=55)
    plt.ylabel('CPU usage (%) ', fontsize=55)
    plt.xlabel('Test Data Record', fontsize=55)
    plt.xticks(xticks)
    
    from matplotlib.legend_handler import HandlerLine2D 
    
    linewidth=6
    def update(handle, orig):
        handle.update_from(orig)
        handle.set_linewidth(linewidth)
    
    #plt.legend()
        
    plt.legend(loc=1, prop={'size': 50}, ncol=4, markerscale=10, scatterpoints=3,handler_map={plt.Line2D : HandlerLine2D(update_func=update)})
        
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(40)
    
    plt.savefig('Experiment 1 t+'+str(m)+'.svg',bbox_inches = 'tight')
    plt.show()