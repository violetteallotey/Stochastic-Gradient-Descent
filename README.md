# Stochastic-Gradient-Descent
Week 12 Individual Project

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml

import matplotlib as mpl
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)
type(mnist)

mnist.keys()

X, y = mnist["data"], mnist["target"]

type(X), type(y)


```python
X.shape, y.shape
```




    ((70000, 784), (70000,))




```python
X.head()
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
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>pixel10</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
y.head()
```




    0    5
    1    0
    2    4
    3    1
    4    9
    Name: class, dtype: category
    Categories (10, object): ['0', '1', '2', '3', ..., '6', '7', '8', '9']




```python
some_digit = X.iloc[0,:].values
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("on")
plt.show()
```


    
![png](output_7_0.png)
    



```python
y = y.astype(np.uint8)
```


```python
y[0]
```




    5




```python
X.isnull().sum().sum() 
```




    0




```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```


```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
#sgd_clf.fit(X_train, y_train_5)
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
sgd_clf.predict([some_digit])
```

    C:\Users\user\anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but SGDClassifier was fitted with feature names
      warnings.warn(
    




    array([3], dtype=uint8)




```python
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
```




    array([0.87365, 0.85835, 0.8689 ])




```python
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```




    array([0.8983, 0.891 , 0.9018])




```python
from sklearn.metrics import confusion_matrix
y_train_pred_sgd = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred_sgd)
conf_mx 
```




    array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
           [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
           [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
           [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
           [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
           [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
           [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
           [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
           [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
           [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],
          dtype=int64)




```python
from sklearn.metrics import classification_report

target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']

print(classification_report(y_train, y_train_pred_sgd,target_names=target_names))
```

                  precision    recall  f1-score   support
    
         class 0       0.97      0.94      0.95      5923
         class 1       0.97      0.95      0.96      6742
         class 2       0.93      0.88      0.90      5958
         class 3       0.92      0.85      0.88      6131
         class 4       0.94      0.89      0.91      5842
         class 5       0.88      0.82      0.85      5421
         class 6       0.95      0.94      0.95      5918
         class 7       0.95      0.91      0.93      6265
         class 8       0.66      0.93      0.77      5851
         class 9       0.90      0.86      0.88      5949
    
        accuracy                           0.90     60000
       macro avg       0.91      0.90      0.90     60000
    weighted avg       0.91      0.90      0.90     60000
    
    


```python
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
```

    C:\Users\user\anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but OneVsOneClassifier was fitted with feature names
      warnings.warn(
    




    array([5], dtype=uint8)




```python
cross_val_score(ovo_clf, X_train, y_train, cv=3, scoring="accuracy")
```




    array([0.91545, 0.9131 , 0.92045])




```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```




    array([0.91595, 0.9149 , 0.91845])




```python
y_train_pred_ovo = cross_val_predict(ovo_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred_ovo)
conf_mx 
```




    array([[5645,    1,   37,   18,    8,  106,   46,    6,   44,   12],
           [   2, 6514,   54,   30,   10,   22,    3,   13,   89,    5],
           [  27,   55, 5351,  152,   84,   37,   61,   52,  124,   15],
           [  13,   22,  122, 5455,    2,  248,    9,   39,  181,   40],
           [   7,   16,   68,    5, 5375,   29,   42,   41,   61,  198],
           [  34,   13,   29,  228,   29, 4820,   61,    7,  163,   37],
           [  35,    9,   76,    9,   41,   85, 5601,    2,   60,    0],
           [   8,   16,   78,   62,   48,   31,    1, 5800,   44,  177],
           [  34,   77,   99,  168,   10,  169,   30,   15, 5196,   53],
           [  13,   34,   37,   64,  180,   57,    1,  249,   85, 5229]],
          dtype=int64)




```python
from sklearn.metrics import classification_report

target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
print(classification_report(y_train, y_train_pred_ovo,target_names=target_names))
```

                  precision    recall  f1-score   support
    
         class 0       0.97      0.95      0.96      5923
         class 1       0.96      0.97      0.97      6742
         class 2       0.90      0.90      0.90      5958
         class 3       0.88      0.89      0.89      6131
         class 4       0.93      0.92      0.92      5842
         class 5       0.86      0.89      0.87      5421
         class 6       0.96      0.95      0.95      5918
         class 7       0.93      0.93      0.93      6265
         class 8       0.86      0.89      0.87      5851
         class 9       0.91      0.88      0.89      5949
    
        accuracy                           0.92     60000
       macro avg       0.92      0.92      0.92     60000
    weighted avg       0.92      0.92      0.92     60000
    
    


```python

```


```python
from scipy.ndimage import interpolation

X_aug_down = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,1,0], cval=0)
X_aug_down.shape
```




    (60000, 28, 28)




```python
X_aug_up = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,-1,0], cval=0)
X_aug_up.shape
```




    (60000, 28, 28)




```python
X_aug_right = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,0,1], cval=0)
X_aug_right.shape
```




    (60000, 28, 28)




```python
X_aug_left = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,0,-1], cval=0)
X_aug_left.shape
```




    (60000, 28, 28)




```python
X_temp = np.concatenate((X_aug_down, X_aug_up, X_aug_right, X_aug_left))
X_temp.shape
```




    (240000, 28, 28)




```python
X_aug = np.concatenate((X_train, X_temp.reshape(240000, 784)))
X_aug.shape
```




    (300000, 784)




```python
y_aug = np.concatenate((y_train, y_train, y_train, y_train, y_train))
y_aug.shape
```




    (300000,)




```python
# Split the train and test set
X_train_shift, X_test_shift, y_train_shift, y_test_shift = X_aug[:300000], X_aug[300000:], y_aug[:300000], y_aug[300000:]
```


```python
from sklearn.linear_model import SGDClassifier

sgd_clf_shift = SGDClassifier(random_state=42)
#sgd_clf.fit(X_train, y_train_5)
sgd_clf_shift.fit(X_train_shift, y_train_shift) # y_train, not y_train_5
sgd_clf_shift.predict([some_digit])
```




    array([5], dtype=uint8)




```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf_shift, X_train_shift, y_train_shift, cv=3, scoring="accuracy")
```




    array([0.84679, 0.79802, 0.81172])




```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled_shift = scaler.fit_transform(X_train_shift.astype(np.float64))
cross_val_score(sgd_clf_shift, X_train_scaled_shift, y_train_shift, cv=3, scoring="accuracy")
```

    C:\Users\user\anaconda3\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:696: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn(
    




    array([0.84325, 0.77361, 0.79561])




```python
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred_sgd_shift = cross_val_predict(sgd_clf_shift, X_train_scaled_shift, y_train_shift, cv=3)
conf_mx = confusion_matrix(y_train_shift, y_train_pred_sgd_shift)
conf_mx 
```


```python
from sklearn.metrics import classification_report

target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']

print(classification_report(y_train_shift, y_train_pred_sgd_shift,target_names=target_names))
```


```python
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train_shift, y_train_shift)
ovo_clf.predict([some_digit])
```

Not completed

## Author
Violette Naa Adoley Allotey