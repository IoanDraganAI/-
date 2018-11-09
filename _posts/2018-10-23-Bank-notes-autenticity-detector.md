---
title: "Bank notes authenticity detector"
categories:
  - DeepLearning
---


We'll use the [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.

The data consists of 5 columns:

* variance of Wavelet Transformed image (continuous)
* skewness of Wavelet Transformed image (continuous)
* curtosis of Wavelet Transformed image (continuous)
* entropy of image (continuous)
* class (integer)

Where class indicates whether or not a Bank Note was authentic.

This sort of task is perfectly suited for Neural Networks and Deep Learning! Just follow the instructions below to get started!

## Get the Data

** Use pandas to read in the bank_note_data.csv file **


```python
import pandas as pd
```


```python
data = pd.read_csv('bank_note_data.csv')
```

** Check the head of the Data **


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image.Var</th>
      <th>Image.Skew</th>
      <th>Image.Curt</th>
      <th>Entropy</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
      <td>-2.8073</td>
      <td>-0.44699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
      <td>-2.4586</td>
      <td>-1.46210</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
      <td>1.9242</td>
      <td>0.10645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
      <td>-4.0112</td>
      <td>-3.59440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
      <td>4.5718</td>
      <td>-0.98880</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## EDA

We'll just do a few quick plots of the data.

** Import seaborn and set matplolib inline for viewing **


```python
import seaborn as sns
%matplotlib inline
```

** Create a Countplot of the Classes (Authentic 1 vs Fake 0) **


```python
sns.countplot(x='Class',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26bb34edda0>



![png]({{ "/assets/img/BankAuth/output_10_1.png" | absolute_url }})



** Create a PairPlot of the Data with Seaborn, set Hue to Class **


```python
sns.pairplot(data,hue='Class')
```

    <seaborn.axisgrid.PairGrid at 0x26bb34c9da0>



![png]({{ "/assets/img/BankAuth/output_12_2.png" | absolute_url }})


## Data Preparation 

When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data, this step isn't actually necessary for our particular data set, but let's run through it for practice!

### Standard Scaling




```python
from sklearn.preprocessing import StandardScaler
```

**Create a StandardScaler() object called scaler.**


```python
scaler = StandardScaler()
```

**Fit scaler to the features.**


```python
scaler.fit(data.drop('Class',axis=1))
```




    StandardScaler(copy=True, with_mean=True, with_std=True)



**Use the .transform() method to transform the features to a scaled version.**


```python
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
```

**Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**


```python
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image.Var</th>
      <th>Image.Skew</th>
      <th>Image.Curt</th>
      <th>Entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.121806</td>
      <td>1.149455</td>
      <td>-0.975970</td>
      <td>0.354561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.447066</td>
      <td>1.064453</td>
      <td>-0.895036</td>
      <td>-0.128767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.207810</td>
      <td>-0.777352</td>
      <td>0.122218</td>
      <td>0.618073</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.063742</td>
      <td>1.295478</td>
      <td>-1.255397</td>
      <td>-1.144029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.036772</td>
      <td>-1.087038</td>
      <td>0.736730</td>
      <td>0.096587</td>
    </tr>
  </tbody>
</table>
</div>



## Train Test Split

** Create two objects X and y which are the scaled feature values and labels respectively.**


```python
X = df_feat
```


```python
y = data['Class']
```

** Use SciKit Learn to create training and testing sets of the data:**


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

# Tensorflow


```python
import tensorflow as tf



** Create a list of feature column objects using tf.feature.numeric_column()**


```python
df_feat.columns
```




    Index(['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy'], dtype='object')




```python
image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')
```


```python
feat_cols = [image_var,image_skew,image_curt,entropy]
```

** Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**


```python
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)
```


** Now create a tf.estimator.pandas_input_fn that takes in your X_train, y_train, batch_size and set shuffle=True. You can play around with the batch_size parameter if you want, but let's start by setting it to 20 since our data isn't very big. **


```python
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)
```

** Now train classifier to the input function. Use steps=500. You can play around with these values if you want!**

*Note: Ignore any warnings you get, they won't effect your output*


```python
classifier.train(input_fn=input_func,steps=500)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into 
    INFO:tensorflow:loss = 13.792015, step = 1
    INFO:tensorflow:Saving checkpoints for 48 into 
    INFO:tensorflow:Loss for final step: 0.47980386.





    <tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x26bbacc7c18>



## Model Evaluation

** Create another pandas_input_fn that takes in the X_test data for x. Remember this one won't need any y_test info since we will be using this for the network to create its own predictions. Set shuffle=False since we don't need to shuffle for predictions.**


```python
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
```

** Use the predict method from the classifier model to create predictions from X_test **


```python
note_predictions = list(classifier.predict(input_fn=pred_fn))
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from 
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.



```python
note_predictions[0]
```




    {'class_ids': array([0], dtype=int64),
     'classes': array([b'0'], dtype=object),
     'logistic': array([0.00157453], dtype=float32),
     'logits': array([-6.4522204], dtype=float32),
     'probabilities': array([0.9984255 , 0.00157453], dtype=float32)}




```python
final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])
```

** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,final_preds))
```

    [[213   2]
     [ 10 187]]



```python
print(classification_report(y_test,final_preds))
```

                 precision    recall  f1-score   support
    
              0       0.96      0.99      0.97       215
              1       0.99      0.95      0.97       197
    
    avg / total       0.97      0.97      0.97       412
    
