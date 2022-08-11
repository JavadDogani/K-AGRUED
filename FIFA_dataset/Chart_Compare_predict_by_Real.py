import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import random


for i in range(3,6):
    print(i)
    data=pd.read_csv("t+"+str(i)+".csv")
    x=data['K-AGRUED']
    y=data['Real']
    date_time_str1 = '1998-07-14 00:00:00'
    now = datetime.datetime.strptime(date_time_str1, '%Y-%m-%d %H:%M:%S')
    
    date_time_str2 = '1998-07-26 21:59:00'
    then = datetime.datetime.strptime(date_time_str2, '%Y-%m-%d %H:%M:%S')
    
    days = mdates.drange(now,then,datetime.timedelta(minutes=1))
    y=data['Real']
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=10) 
    plt.plot(days,y,'b-', label='Real Value',markersize=8)
    plt.plot(days,x,'-r', label='Predicted Value',markersize=8)
    plt.gcf().autofmt_xdate()
    plt.gcf().set_size_inches(12, 5)
    plt.title('FIFA Dataset (t+'+str(i)+')', fontsize=14)
    plt.xlabel('Time ', fontsize=14)
    plt.ylabel('Workload (HTTP Request)', fontsize=14)
    plt.legend(loc=1, prop={'size': 15}, ncol=8, markerscale=4., scatterpoints=3,)
   
    plt.savefig('Compare_by_Real on t'+str(i)+'.svg',bbox_inches = 'tight')
    plt.show()