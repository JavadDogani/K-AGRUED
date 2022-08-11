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

d1=pd.read_csv("nasa.csv")
data=d1
x=data['period']
y=data['count'][:data.shape[0]]
date_time_str1 = x[0]
now = datetime.datetime.strptime(date_time_str1, '%Y-%m-%d %H:%M:%S')

date_time_str2 = '1995-07-28 13:32:00'
then = datetime.datetime.strptime(date_time_str2, '%Y-%m-%d %H:%M:%S')

now1 = datetime.datetime.strptime(date_time_str1, '%Y-%m-%d %H:%M:%S')

date_time_str2 = '1995-08-01 00:00:00'
then1 = datetime.datetime.strptime(date_time_str2, '%Y-%m-%d %H:%M:%S')


days1 = mdates.drange(now,then,datetime.timedelta(minutes=1))
days2=mdates.drange(now1,then1,datetime.timedelta(minutes=1))
y=data['count'][:days1.shape[0]+days2.shape[0]]
days=np.concatenate((days1,days2), axis=0)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
#fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
plt.plot(days,y)
plt.gcf().autofmt_xdate()
plt.gcf().set_size_inches(14.5, 4.5)
plt.title('NASA Dataset', fontsize=18)
plt.xlabel('Time (Second)', fontsize=16)
plt.ylabel('Workload (HTTP Request)', fontsize=16)

plt.savefig('NASA_data.svg',bbox_inches = 'tight')
plt.show()