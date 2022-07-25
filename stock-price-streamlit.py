import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0, 1))

st.subheader('----------------------------------------------------------')

#-----------------------------------------------------------------------------------------------------------

st.title("Stock Price Prediction Model")
st.caption("by Kripa Nath")

user_input=st.text_input("Enter Ticker Symbol from Yahoo Finance",'AMZN')
df=yf.download(user_input,period='max')
df=pd.DataFrame(df)
st.subheader('----------------------------------------------------------')

#-----------------------------------------------------------------------------------------------------------

st.subheader('Data')
st.write(df)
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Genric Observations')
st.write('Total Columns:',df.shape[1])
st.write('Total Data Points for Each Column:',df.shape[0])
st.write('First Data Point from:',df.index[0])
st.write('Last Data Point from:',df.index[-1])
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Some Statistical Summary')
st.write(df.describe())
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Plot of Each Category - Open, High, Low, Close, Adj. Close, Volume with Time')
feat_list=df.columns
f1 = st.selectbox('Select the feature:', feat_list)
f2 = st.selectbox('Select the time-period:', ['Daily','Monthly(Average)','Yearly(Average)'])
fig=plt.figure(figsize=(15,8))
if f2=='Daily' :
    values = st.slider('Select the range',0,len(df), (0,len(df)))
    plt.plot(df.index[values[0]:values[1]], df[f1][values[0]:values[1]])
elif f2=='Monthly(Average)':
    plt.plot(df.resample(rule='BM').mean()[f1])
else:
    plt.plot(df.resample(rule='BY').mean()[f1])
st.pyplot(fig)
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Pair Plot between Each Features')
feat_list=df.columns
f1 = st.selectbox('Select the first(x-value) feature:', feat_list)
f2 = st.selectbox('Select the second(y-value) feature:', feat_list)
fig=plt.figure(figsize=(15,8))
plt.scatter(df[f1],df[f2])
st.pyplot(fig)
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Features vs Time with Moving Average 50, 100 & 200')
f1 = st.selectbox('Select the feature:', feat_list,3)
ma50=df[f1].rolling(50).mean()
ma100=df[f1].rolling(100).mean()
ma200=df[f1].rolling(200).mean()
fig= plt.figure(figsize=(18,10))
plt.plot(df[f1])
plt.plot(ma50, color='red')
plt.plot(ma100, color='yellow')
plt.plot(ma200, color='green')
plt.legend([f1,'MA50','MA100','MA200'])
st.pyplot(fig)
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('Linear Models for Price Prediction - Training and Results')

st.write("Training the model using data from 2006 to 2016.")
st.write("Testing the model on data of 2017 onwards.")
st.write("Model Approach: Using previous 100 stock price to predict the current price.")




#data-preprocessing

st.write("Note: In some of the stock data, prices were very low in intial 5-10 years therefore please enter a suitable start DATE after looking at the above graphs which will help in feature selection and modelling ")

date=st.text_input("Enter a starting date (YYYY-MM-DD)",df.index[1200])

feature= st.selectbox('Select the feature you want to model:', feat_list[:4],3)

df1=df.loc[date:]
data=pd.DataFrame(df1[feature])
for i in range(1,101):
    string= feature + ' '+ '{}'
    data[string.format(i)]= data[feature].shift(+(101-i))
data.dropna(axis=0,inplace=True)
colm=data.columns
st.write('Generated Features (first col is target, rest are features)',data.shape)
st.write(data)

split= st.number_input('Enter the size of training data (For ex: If 60%, Enter 60):',60)

model_data=data.loc[data.index[0]:]
t=len(model_data)/100

train=data[:int(t*split)]
test=data[int(t*split):]
st.write('Training Data:',train.shape, train)
st.write('Test Data',test.shape, test)


#--------------------------------------

#training data

train=pd.DataFrame(scaler.fit_transform(train),columns=train.columns,index=train.index)
y_train=train[feature].values
train.drop(feature,axis=1,inplace=True)
X_train=train.values

#--------------------------------------

#testing data

y_test=pd.DataFrame(test[feature])
test.drop(feature,axis=1,inplace=True)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)
X_test=test.values

#--------------------------------------

#training model
result0=st.button('Click to train the Linear Model')
if result0:
    Lr,Rd=LinearRegression(),Ridge()
    Lr,Rd=Lr.fit(X_train,y_train), Rd.fit(X_train,y_train)
    y_testt=scaler.fit_transform(y_test)

    st.write("R^2 for Linear Regression----",Lr.score(X_test,y_testt))
    st.write("R^2 for Linear Regression----",Rd.score(X_test,y_testt))

    #--------------------------------------

    y_lr=Lr.predict(X_test)
    y_rd=Rd.predict(X_test)
    y_lr=scaler.inverse_transform([y_lr])
    y_rd=scaler.inverse_transform([y_rd])

    #--------------------------------------
    st.subheader('Predicted Results')

    fig=plt.figure(figsize=(15,8))
    plt.plot(data[feature],'tab:blue')
    plt.plot(data.index[X_train.shape[0]:],y_lr[0],'indianred')
    plt.plot(data.index[X_train.shape[0]:],y_rd[0],'chocolate')
    plt.axvline(x=data.index[X_train.shape[0]],color='lightcoral')
    plt.axvline(x=data.index[-1],color='lightcoral')

    plt.legend(['Original','Predicted Test Result via Linear Regression','Predicted Test Result via Ridge Regression'])
    st.pyplot(fig)

    #--------------------------------------

    fig=plt.figure(figsize=(15,8))
    plt.plot(data.index[X_train.shape[0]:],y_test.values,'tab:blue')
    plt.plot(data.index[X_train.shape[0]:],y_lr[0],'indianred')
    plt.plot(data.index[X_train.shape[0]:],y_rd[0],'chocolate')
    plt.axvline(x=data.index[X_train.shape[0]],color='lightcoral')
    plt.axvline(x=data.index[-1],color='lightcoral')
    plt.legend(['Original','Predicted by Linear Regression','Predicted by Ridge Regression'])
    st.pyplot(fig)
st.subheader('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

st.subheader('LSTM Models for Close Price Prediction - Training and Results')

st.write("Training the model using data from 2006 to 2016.")
st.write("Validating the model on data from 2017 to 2019.")
st.write("Testing the model on data of 2020 onwards.")
st.write("Model Approach: Using previous 100 stock price to predict the current price.")


#data-preprocessing

date=st.text_input("Enter a starting date (YYYY-MM-DD)",df.index[1200],key=2)

feature= st.selectbox('Select the feature you want to model:', feat_list[:4],0)

df1=df.loc[date:]
data=pd.DataFrame(df1[feature])
for i in range(1,101):
    string= feature + ' '+ '{}'
    data[string.format(i)]= data[feature].shift(+(101-i))
data.dropna(axis=0,inplace=True)
colm=data.columns
st.write('Generated Features (first col is target, rest are features)')
st.write(data)

split0= st.number_input('Enter the size of training data (For ex: If 60%, Enter 60):',60,key=0)
split1= st.number_input('Enter the size of validation data (For ex: If 15%, Enter 15):',15,key=1)


model_data=data.loc[data.index[0]:]
t=len(model_data)/100
t1=len(model_data)/100

train=data[:int(t*split0)]
val=data[int(t*split0):int(t*split0)+int(t1*split1)]
test=data[int(t*split0)+int(t1*split1):]
st.write('Training Data:',train.shape, train)
st.write('Validation Data',val.shape, val)
st.write('Test Data',test.shape, test)


#--------------------------------------

#training data

train=pd.DataFrame(scaler.fit_transform(train),columns=train.columns,index=train.index)
y_train=train[feature].values
train.drop(feature,axis=1,inplace=True)
X_train=train.values

#--------------------------------------

#validation data

val=pd.DataFrame(scaler.fit_transform(val),columns=val.columns,index=val.index)
y_val=val[feature].values
val.drop(feature,axis=1,inplace=True)
X_val=val.values

#--------------------------------------

#testing data

y_test=pd.DataFrame(test[feature])
test.drop(feature,axis=1,inplace=True)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)
X_test=test.values

#--------------------------------------

#training model

model=Sequential()
model.add(LSTM(200,activation='tanh',return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.1))
model.add(LSTM(200,activation='tanh',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(250,activation='tanh',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(250,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.0001),loss='mean_squared_error',metrics='mean_squared_error')
model.summary()

#--------------------------------------

ep=st.selectbox('For how many epocs you want to train the model?', [2,5,10,20,30] ,0)
result=st.button('Click to train the LSTM Model')
if result:

    #--------------------------------------
    
    history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=ep)
    fig=plt.figure(figsize=(15,8))
    plt.style.use("ggplot")
    plt.plot(history.history['loss'], color='b', label="Training loss")
    plt.plot(history.history['val_loss'], color='r', label="Validation loss")
    plt.legend()
    st.pyplot(fig)
    
    #--------------------------------------
    
    y_pred=model.predict(X_test)
    y_testt=scaler.fit_transform(y_test)
    y_lstm=scaler.inverse_transform(y_pred)
    st.write("R^2 for LSTM Model----",r2_score(y_pred,y_testt))
    
    
    #--------------------------------------
    st.subheader('Predicted Results')
    
    fig=plt.figure(figsize=(18,12))
    plt.plot(data[feature],'tab:blue')
    plt.plot(data.index[X_train.shape[0]+X_val.shape[0]:],y_lstm,'indianred')
    plt.axvline(x=data.index[X_train.shape[0]+X_val.shape[0]],color='lightcoral')
    plt.axvline(x=data.index[-1],color='lightcoral')
    plt.legend(['Original','Predicted Test Result via LSTM'])
    st.pyplot(fig)
    
    #--------------------------------------

    fig=plt.figure(figsize=(18,10))
    plt.plot(data.index[X_train.shape[0]+X_val.shape[0]:],y_test.values,'tab:blue')
    plt.plot(data.index[X_train.shape[0]+X_val.shape[0]:],y_lstm,'indianred')
    plt.axvline(x=data.index[X_train.shape[0]+X_val.shape[0]],color='lightcoral')
    plt.axvline(x=data.index[-1],color='lightcoral')
    plt.legend(['Original','Predicted by LSTM'])
    st.pyplot(fig)
    
st.subheader('----------------------------------------------------------')









