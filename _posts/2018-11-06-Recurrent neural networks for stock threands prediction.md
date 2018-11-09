---
title: "Bank notes authenticity detector"
categories:
  - DeepLearning
---

In this project I will use Recurrent Neural network to predict the threand in the Google stock price.

Thein thr model on 5 years stock price and try to predict 2017.

# Part 1 - Data Preprocessing

Importing the libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Importing the training set


```python
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
```

Feature Scaling


```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

Creating a data structure with 60 timesteps and 1 output


```python
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
```

Reshaping


```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

# Building the RNN

Importing the Keras libraries and packages


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
```

Initialising the RNN


```python
regressor = Sequential()
```

Adding the first LSTM layer and some Dropout regularisation


```python
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
```

Adding a second LSTM layer and some Dropout regularisation


```python
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
```

Adding a third LSTM layer and some Dropout regularisation


```python
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
```

Adding a fourth LSTM layer and some Dropout regularisation


```python
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
```

Adding the output layer


```python
regressor.add(Dense(units = 1))
```

Compiling the RNN


```python
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

Fitting the RNN to the Training set


```python
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```

    Epoch 1/100
    1198/1198 [==============================] - 14s 12ms/step - loss: 0.0568
    Epoch 2/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0062
    Epoch 3/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0055
    Epoch 4/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0046
    Epoch 5/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0050
    Epoch 6/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0048
    Epoch 7/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0050
    Epoch 8/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0044
    Epoch 9/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0045
    Epoch 10/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0043
    Epoch 11/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0039
    Epoch 12/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0047
    Epoch 13/100
    1198/1198 [==============================] - 11s 10ms/step - loss: 0.0039
    Epoch 14/100
    1198/1198 [==============================] - 11s 9ms/step - loss: 0.0038
    Epoch 15/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0039
    Epoch 16/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0039
    Epoch 17/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0038
    Epoch 18/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0038
    Epoch 19/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0039
    Epoch 20/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0035
    Epoch 21/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0032
    Epoch 22/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0033
    Epoch 23/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0036
    Epoch 24/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0034
    Epoch 25/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0033
    Epoch 26/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0034
    Epoch 27/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0029
    Epoch 28/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0033
    Epoch 29/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0030
    Epoch 30/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0031
    Epoch 31/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0029
    Epoch 32/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0028
    Epoch 33/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0027
    Epoch 34/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0028
    Epoch 35/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0025
    Epoch 36/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0030
    Epoch 37/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 38/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 39/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0027
    Epoch 40/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0027
    Epoch 41/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0027
    Epoch 42/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0027
    Epoch 43/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0029
    Epoch 44/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0029
    Epoch 45/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 46/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0025
    Epoch 47/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 48/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0026
    Epoch 49/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0023
    Epoch 50/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 51/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0023
    Epoch 52/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 53/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0022
    Epoch 54/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0022
    Epoch 55/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0023
    Epoch 56/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 57/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0020
    Epoch 58/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 59/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0024
    Epoch 60/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 61/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0020
    Epoch 62/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 63/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 64/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 65/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 66/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 67/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0022
    Epoch 68/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 69/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 70/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 71/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 72/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 73/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0017
    Epoch 74/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0019
    Epoch 75/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 76/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 77/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0017
    Epoch 78/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 79/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0017
    Epoch 80/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 81/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 82/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 83/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0017
    Epoch 84/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0021
    Epoch 85/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0017
    Epoch 86/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0018
    Epoch 87/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 88/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 89/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 90/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 91/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 92/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0016
    Epoch 93/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 94/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0013
    Epoch 95/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0013
    Epoch 96/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 97/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 98/100
    1198/1198 [==============================] - 11s 10ms/step - loss: 0.0013
    Epoch 99/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015
    Epoch 100/100
    1198/1198 [==============================] - 12s 10ms/step - loss: 0.0015





    <keras.callbacks.History at 0x7ff24469e860>



# Making the predictions and visualising the results

Getting the real stock price of 2017


```python
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
```

Getting the predicted stock price of 2017


```python
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

Visualising the results


```python
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```


![png](output_37_0.png)

