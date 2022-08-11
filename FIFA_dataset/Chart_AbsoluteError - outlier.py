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

def AbsoluteError(predictions, targets):
    Abserror=[]
    length=len(targets)
    for i in range(length):
        Abserror.append(abs(predictions[i]-targets[i]))
    return Abserror

lbl=[ 'ARIMA', 'LSTM', 'GRUED', 'K-AGRUED']
colors = [ 'lightcoral','lightsalmon','gold','yellowgreen']
for time in range(3,6):
    dataset=[]
    AbsList=np.array([])
    realarr = np.array([])
    inputfilepath_predict="t+"+str(time)+".csv"
    x=pd.read_csv(inputfilepath_predict)
    data=pd.read_csv(inputfilepath_predict)["Real"].values
    realarr=np.append(realarr,data)
    for j in lbl:
        feildvalue = np.array([])    
        data=x[j].values
        feildvalue=np.append(feildvalue,data)
        ABSE=AbsoluteError(realarr, feildvalue)
        dataset.append(ABSE)       
    
    
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    bplot1 = ax1.boxplot(dataset,
                         notch=False,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=lbl,  flierprops={'marker': 'o', 'markersize': 4})  # will be used to label x-ticks
    
    
    for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)
    # adding horizontal grid lines
    for ax in [ax1]:
        ax.yaxis.grid(True)
        #ax.set_xlabel('Three separate samples', fontsize=24)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('FIFA Dataset (t+'+str(time)+')', fontsize=12)
    
       
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    	label.set_fontsize(10)
    
    plt.ylim([-10, 20000])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)
    for label in ( ax.get_yticklabels()):
        label.set_fontsize(8)
    plt.savefig('Absolute-outlier t+'+str(time)+'.svg',bbox_inches = 'tight')
    plt.show()


