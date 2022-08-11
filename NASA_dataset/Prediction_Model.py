from __future__ import print_function
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
#Initial Variable
Iteration=20
import math
lagNumber=2
def addColumns(data):
    for i in range(1,3):
        data['lag_'+str(i)] = data['count'].shift(i)

 

def LSTM(X_train,Y_train,X_test,Y_test,Epoch,featurenumber, scalar2,scalar4):
    x_train=X_train.copy()
    x_test=X_test.copy()
    y_train=Y_train.copy()
    y_test=Y_test.copy()
    
    import numpy
    from keras.models import Sequential
    from keras.layers import Dense,LSTM
   
    col=x_train.shape[1]
    #x_train = numpy.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
    x_train = numpy.reshape(x_train, (x_train.shape[0],1, x_train.shape[1]))
    #x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    x_test = numpy.reshape(x_test, (x_test.shape[0],1, x_test.shape[1]))  
    from keras.losses import categorical_crossentropy
    
    look_back=featurenumber
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=Epoch, batch_size=1, verbose=2)
    
    import math
    from sklearn.metrics import mean_squared_error
     # make predictions
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    # invert predictions
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0, 1))
    scaler=scalar2
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform(y_train)
    scaler=scalar4    
    testPredict1 = scaler.inverse_transform(testPredict)
    y_test = scaler.inverse_transform(y_test)
    r1=MAPE( trainPredict,y_train)
    r2=MAPE(testPredict1,y_test)
    
    return r1,r2,trainPredict,testPredict1

def RMSE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        sum+=abs(predictions[i]- targets[i])**2
    return math.sqrt(sum/length)

def MAPE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        if (targets[i]!=0):
            sum+=abs((predictions[i] - targets[i]))/targets[i]
    return sum/length




def MAE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        sum+=abs(predictions[i]- targets[i])
    return sum/length

def run(data):
    loc=int(0.7*data.shape[0])    
    train=data[0:loc]
    train_y=train['count']
    train_x=train.drop(columns={'count'})    
    test=data[loc:]
    test_y=test['count']
    test_x=test.drop(columns={'count'})
    
    
    
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    #dt = scaler1.fit_transform(target.values.reshape(-1,1)
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler3 = MinMaxScaler(feature_range=(0, 1))
    scaler4 = MinMaxScaler(feature_range=(0, 1))
    
    x_train_normal = scaler1.fit_transform(train_x.values)
    y_train_normal = scaler2.fit_transform(train_y.values.reshape(-1,1))
    x_test_normal = scaler3.fit_transform(test_x.values)
    y_test_normal = scaler4.fit_transform(test_y.values.reshape(-1,1))
    
    
    featurenumber=train_x.shape[1]
    MAPE_train,MAPE_test,pred_train,pred_test=LSTM(x_train_normal,y_train_normal,x_test_normal,y_test_normal,Iteration,featurenumber,scaler2,scaler4)
    
    Result_MAPE=MAPE(pred_test,test_y.values)
    
    
        
    #print("MAPE:",Result_MAPE)
    print("MAPE_train:",MAPE_train)
    #print("MAPE_test:",MAPE_test)
    Result_RMSE=RMSE(pred_test,test_y.values)
    Result_MAE=MAE(pred_test,test_y.values)
    Result_MAPE=MAPE(pred_test,test_y.values)   
    print(" Test\n MAPE \t\t\t RMSE \t\t\t MAE:")
    print( Result_MAPE,"\t",Result_RMSE,"\t",Result_MAE)
    
    
       
    return test_y.values,pred_test

def preprocessing(corr):
    df=pd.DataFrame()
    df['Date']=corr.index
    df['count']=corr.values
    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d %H:%M:%S')  
    #df['month']=df['Date'].dt.month
    df['day']=df['Date'].dt.day
    df['dayofweek']=df['Date'].dt.dayofweek
    df['Hour']=df['Date'].dt.hour 
    df['minute']=df['Date'].dt.minute 
    for i in range(0,lagNumber):
        df['lag_'+str(i+1)] = df['count'].shift(i+1)
    for j in range(lagNumber):
        for i in range(0,lagNumber):
            df['lag_'+str(i+1)].values[j] = df['count'].values[j]
    df.drop(columns={'Date'},inplace=True)
    return df


data=pd.read_csv("NASA_Results.csv", index_col=0, parse_dates=True)

data=preprocessing(data)
addColumns(data)
data=data[lagNumber:]
test_data,pred_data=run(data)

