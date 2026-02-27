
#                                       bitcoin stock price prediction --------------
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

#download data

btc = yf.Ticker("BTC-USD")
btc_data = btc.history(period="max")

#clean data 

del btc_data["Dividends"]
del btc_data["Stock Splits"]

print(btc_data.head())


#Preprocess Data
ma_50_days = btc_data.Close.rolling(50).mean()
ma_100_days = btc_data.Close.rolling(100).mean()

plt.figure(figsize= (8,7))
plt.plot(ma_50_days,"r")
plt.plot(ma_100_days,"b")
plt.plot(btc_data.Close,"g")
plt.show()
#drop nan values
btc_data = btc_data.dropna()

#data split---
split = int(len(btc_data) * 0.80)

data_train = btc_data[:split]
data_test  = btc_data[split:]


scaler = MinMaxScaler(feature_range=(0,1))
train_scaler = scaler.fit_transform(data_train[['Close']])

x = []
y = []

for i in range(50, train_scaler.shape[0]):
    x.append(train_scaler[i-50:i])
    y.append(train_scaler[i,0])

#creat model

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(LSTM(units = 50 , activation= "relu", return_sequences = True,
               input_shape = ((x.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(units= 60, activation="relu", return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units= 80, activation= "relu", return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units= 120, activation= "relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer = "adam", loss = "mean_squared_error")
model.fit(x,y, epochs= 50 ,batch_size= 32, verbose= 1)

model.summary()

pst_50_days = data_train.tail(50)

data_test = pd.concat([pst_50_days, data_test], ignore_index= True)
test_scaler = scaler.fit_transform(data_test[['Close']])

x = []
y = []

for i in range(50, test_scaler.shape[0]):
    x.append(test_scaler[i-50:i])
    y.append(test_scaler[i,0])

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], x.shape[1], 1)

y_predict = model.predict(x)

scale = 1/scaler.scale_
y_predict = y_predict*scale
y = y*scale

#Visualize the Results

plt.figure(figsize=(10,8))
plt.plot(y_predict, "r", label = "predicted price")
plt.plot(y, "g", label = "original  price")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.show()
model.save("stock_model.keras")

