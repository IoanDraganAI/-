---
title: "Convolutional neural networks for image processing"
categories:
  - DeepLearning
---

In this project I will use convolutional neural networks for image classification.

I will work on images of cats/dogs and try to predict wich class each image belongs to. 

# Building the CNN

Importing the Keras libraries and packages


```python
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
```

Initialising the CNN


```python
classifier = Sequential()
```

Convolution


```python
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
```

Pooling


```python
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```

Adding a second convolutional layer


```python
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```

Flattening


```python
classifier.add(Flatten())
```

Full connection


```python
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
```

Compiling the CNN


```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

# Fitting the CNN to the images

Importing Keras image augmentation library to enrich data set and reduce overfitting


```python
from keras.preprocessing.image import ImageDataGenerator
```

Apply image transformation on the training set


```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
```

Apply image transformation to rescale the training set


```python
test_datagen = ImageDataGenerator(rescale = 1./255)
```

Create the training set


```python
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
```

    Found 8000 images belonging to 2 classes.


Create the test set


```python
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
```

    Found 2000 images belonging to 2 classes.


Fit the CNN on the training set and test in on the test set


```python
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
```

    Epoch 1/25
    8000/8000 [==============================] - 1244s 156ms/step - loss: 0.3651 - acc: 0.8276 - val_loss: 0.5617 - val_acc: 0.8062
    Epoch 2/25
    8000/8000 [==============================] - 1157s 145ms/step - loss: 0.1144 - acc: 0.9563 - val_loss: 0.8215 - val_acc: 0.8118
    Epoch 3/25
    8000/8000 [==============================] - 1038s 130ms/step - loss: 0.0549 - acc: 0.9801 - val_loss: 0.9595 - val_acc: 0.8224
    Epoch 4/25
    8000/8000 [==============================] - 1074s 134ms/step - loss: 0.0370 - acc: 0.9870 - val_loss: 1.0793 - val_acc: 0.8138
    Epoch 5/25
    8000/8000 [==============================] - 958s 120ms/step - loss: 0.0301 - acc: 0.9895 - val_loss: 1.2260 - val_acc: 0.8152
    Epoch 6/25
    8000/8000 [==============================] - 912s 114ms/step - loss: 0.0245 - acc: 0.9917 - val_loss: 1.3117 - val_acc: 0.8137
    Epoch 7/25
    8000/8000 [==============================] - 889s 111ms/step - loss: 0.0217 - acc: 0.9925 - val_loss: 1.3298 - val_acc: 0.8171
    Epoch 8/25
    8000/8000 [==============================] - 888s 111ms/step - loss: 0.0195 - acc: 0.9936 - val_loss: 1.3475 - val_acc: 0.8079
    Epoch 9/25
    8000/8000 [==============================] - 899s 112ms/step - loss: 0.0179 - acc: 0.9942 - val_loss: 1.4131 - val_acc: 0.7971
    Epoch 10/25
    8000/8000 [==============================] - 894s 112ms/step - loss: 0.0159 - acc: 0.9947 - val_loss: 1.3990 - val_acc: 0.8119
    Epoch 11/25
    8000/8000 [==============================] - 892s 112ms/step - loss: 0.0154 - acc: 0.9953 - val_loss: 1.3948 - val_acc: 0.8106
    Epoch 12/25
    8000/8000 [==============================] - 895s 112ms/step - loss: 0.0139 - acc: 0.9955 - val_loss: 1.5420 - val_acc: 0.8060
    Epoch 13/25
    8000/8000 [==============================] - 999s 125ms/step - loss: 0.0125 - acc: 0.9959 - val_loss: 1.6156 - val_acc: 0.7987
    Epoch 14/25
    8000/8000 [==============================] - 1134s 142ms/step - loss: 0.0118 - acc: 0.9963 - val_loss: 1.5096 - val_acc: 0.8078
    Epoch 15/25
    8000/8000 [==============================] - 835s 104ms/step - loss: 0.0115 - acc: 0.9963 - val_loss: 1.5755 - val_acc: 0.8027
    Epoch 16/25
    8000/8000 [==============================] - 834s 104ms/step - loss: 0.0111 - acc: 0.9965 - val_loss: 1.7055 - val_acc: 0.8060
    Epoch 17/25
    8000/8000 [==============================] - 834s 104ms/step - loss: 0.0096 - acc: 0.9969 - val_loss: 1.5730 - val_acc: 0.8225
    Epoch 18/25
    8000/8000 [==============================] - 877s 110ms/step - loss: 0.0100 - acc: 0.9969 - val_loss: 1.6430 - val_acc: 0.8080
    Epoch 19/25
    8000/8000 [==============================] - 1113s 139ms/step - loss: 0.0087 - acc: 0.9972 - val_loss: 1.6590 - val_acc: 0.7944
    Epoch 20/25
    8000/8000 [==============================] - 1081s 135ms/step - loss: 0.0089 - acc: 0.9974 - val_loss: 1.6444 - val_acc: 0.8144
    Epoch 21/25
    8000/8000 [==============================] - 861s 108ms/step - loss: 0.0091 - acc: 0.9973 - val_loss: 1.6282 - val_acc: 0.8071
    Epoch 22/25
    8000/8000 [==============================] - 892s 111ms/step - loss: 0.0084 - acc: 0.9974 - val_loss: 1.7206 - val_acc: 0.8022
    Epoch 23/25
    8000/8000 [==============================] - 990s 124ms/step - loss: 0.0074 - acc: 0.9977 - val_loss: 1.8200 - val_acc: 0.7964
    Epoch 24/25
    8000/8000 [==============================] - 897s 112ms/step - loss: 0.0070 - acc: 0.9978 - val_loss: 1.6997 - val_acc: 0.7970
    Epoch 25/25
    8000/8000 [==============================] - 898s 112ms/step - loss: 0.0081 - acc: 0.9976 - val_loss: 1.7383 - val_acc: 0.8077





    <keras.callbacks.History at 0x7f64f7fe47b8>



# Making new predictions on single images

Importing numpy dimension expansion library and keras image library


```python
import numpy as np
from keras.preprocessing import image
```

Expand the dimensionality of the input image (the predict method expects inputs in a batch):


```python
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
```

Predict the class of a given input


```python
training_set.class_indices
```




    {'cats': 0, 'dogs': 1}




```python
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
```


```python
print(prediction)
```

    dog

