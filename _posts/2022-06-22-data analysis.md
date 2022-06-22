# Real data analysis (when confounder is considered)

### Notice

Before you implement below codes, you should confirm that R is installed in your PC.


```python
!pip install packages rpy2 
```

    Requirement already satisfied: packages in c:\programdata\anaconda3\lib\site-packages (0.1.0)
    Requirement already satisfied: rpy2 in c:\programdata\anaconda3\lib\site-packages (3.4.5)
    Requirement already satisfied: cffi>=1.10.0 in c:\programdata\anaconda3\lib\site-packages (from rpy2) (1.14.5)
    Requirement already satisfied: pytz in c:\programdata\anaconda3\lib\site-packages (from rpy2) (2021.1)
    Requirement already satisfied: tzlocal in c:\programdata\anaconda3\lib\site-packages (from rpy2) (4.0.1)
    Requirement already satisfied: jinja2 in c:\programdata\anaconda3\lib\site-packages (from rpy2) (2.11.3)
    Requirement already satisfied: pycparser in c:\programdata\anaconda3\lib\site-packages (from cffi>=1.10.0->rpy2) (2.20)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\programdata\anaconda3\lib\site-packages (from jinja2->rpy2) (1.1.1)
    Requirement already satisfied: tzdata in c:\programdata\anaconda3\lib\site-packages (from tzlocal->rpy2) (2021.4)
    Requirement already satisfied: backports.zoneinfo in c:\programdata\anaconda3\lib\site-packages (from tzlocal->rpy2) (0.2.1)
    Requirement already satisfied: pytz-deprecation-shim in c:\programdata\anaconda3\lib\site-packages (from tzlocal->rpy2) (0.1.0.post0)
    


```python
import uiat # it will takes some time.
```


```python
import numpy as np
import pandas as pd
```

# EDA


```python
df=pd.read_csv("forestfires.csv")
```


```python
import matplotlib.pyplot as plt

plt.scatter(df['wind'],df['area'])
```




    <matplotlib.collections.PathCollection at 0x22c84310>




    
![png](output_7_1.png)
    



```python
df.describe()
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
      <th>X</th>
      <th>Y</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
      <td>517.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.669246</td>
      <td>4.299807</td>
      <td>90.644681</td>
      <td>110.872340</td>
      <td>547.940039</td>
      <td>9.021663</td>
      <td>18.889168</td>
      <td>44.288201</td>
      <td>4.017602</td>
      <td>0.021663</td>
      <td>12.847292</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.313778</td>
      <td>1.229900</td>
      <td>5.520111</td>
      <td>64.046482</td>
      <td>248.066192</td>
      <td>4.559477</td>
      <td>5.806625</td>
      <td>16.317469</td>
      <td>1.791653</td>
      <td>0.295959</td>
      <td>63.655818</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>18.700000</td>
      <td>1.100000</td>
      <td>7.900000</td>
      <td>0.000000</td>
      <td>2.200000</td>
      <td>15.000000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>90.200000</td>
      <td>68.600000</td>
      <td>437.700000</td>
      <td>6.500000</td>
      <td>15.500000</td>
      <td>33.000000</td>
      <td>2.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>91.600000</td>
      <td>108.300000</td>
      <td>664.200000</td>
      <td>8.400000</td>
      <td>19.300000</td>
      <td>42.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.520000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>5.000000</td>
      <td>92.900000</td>
      <td>142.400000</td>
      <td>713.900000</td>
      <td>10.800000</td>
      <td>22.800000</td>
      <td>53.000000</td>
      <td>4.900000</td>
      <td>0.000000</td>
      <td>6.570000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>96.200000</td>
      <td>291.300000</td>
      <td>860.600000</td>
      <td>56.100000</td>
      <td>33.300000</td>
      <td>100.000000</td>
      <td>9.400000</td>
      <td>6.400000</td>
      <td>1090.840000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.hist(df['wind'])
```




    (array([ 14.,  45.,  97.,  93.,  92., 113.,  19.,  26.,   5.,  13.]),
     array([0.4, 1.3, 2.2, 3.1, 4. , 4.9, 5.8, 6.7, 7.6, 8.5, 9.4]),
     <BarContainer object of 10 artists>)




    
![png](output_9_1.png)
    



```python
plt.hist(df['area'])
```




    (array([508.,   6.,   1.,   0.,   0.,   0.,   1.,   0.,   0.,   1.]),
     array([   0.   ,  109.084,  218.168,  327.252,  436.336,  545.42 ,
             654.504,  763.588,  872.672,  981.756, 1090.84 ]),
     <BarContainer object of 10 artists>)




    
![png](output_10_1.png)
    



```python
plt.hist(np.log(df['area']+1))
```




    (array([275.,  61.,  59.,  54.,  29.,  21.,   9.,   6.,   1.,   2.]),
     array([0.        , 0.69956196, 1.39912393, 2.09868589, 2.79824785,
            3.49780981, 4.19737178, 4.89693374, 5.5964957 , 6.29605766,
            6.99561963]),
     <BarContainer object of 10 artists>)




    
![png](output_11_1.png)
    


# Ignorability Assumption Test


```python
df.loc[df["month"]=="jan","month"]=1
df.loc[df["month"]=="feb","month"]=2
df.loc[df["month"]=="mar","month"]=3
df.loc[df["month"]=="apr","month"]=4
df.loc[df["month"]=="may","month"]=5
df.loc[df["month"]=="jun","month"]=6
df.loc[df["month"]=="jul","month"]=7
df.loc[df["month"]=="aug","month"]=8
df.loc[df["month"]=="sep","month"]=9
df.loc[df["month"]=="oct","month"]=10
df.loc[df["month"]=="nov","month"]=11
df.loc[df["month"]=="dec","month"]=12

A = np.array(df["wind"])
Y = np.array(df["area"])
Z = np.array(df["month"])
X = np.array(df[["X","Y","temp","RH","rain"]])


UIAT = uiat.UniversalIgnorabilityAssumptionTest(cause=A,effect=Y,explorer=Z,covariates=X,dtype="continuous",verbose=False)
pvalue = UIAT.test()
print(pvalue) # conclude no violation of ignorability assumption
```

    0.21007211378928048
    

# Causal Inference using Machine Learning


```python
X,y = np.array(df[["X","Y","wind","temp","RH","rain"]]), np.array(df["area"])
```


```python
np.random.seed(1)
train_idx=np.random.choice(517,517,replace=False)
```


```python
X_train = X[train_idx,]
y_train = y[train_idx,]
```


```python
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics   #Additional scklearn functions

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
```


```python
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
# the fine-tuned XGBoost
xgbt_regr = xgb.XGBRegressor(
    learning_rate=0.01,
    n_estimators=278,
    max_depth=1,
    min_child_weight=7,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.7,
    objective='reg:squarederror',
    nthread=3,
    scale_pos_weight=1,
    reg_alpha=0.01,
    reg_lambda=1,
    seed=1
    )
```


```python
train=df.iloc[train_idx,]
train.reset_index(drop=True,inplace=True)

train['area']=np.log(train['area']+1)
target = 'area'
predictors = ["X","Y","wind","temp","RH","rain"]
```


```python
xgbt_regr.fit(train[predictors],train[target])
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.7, enable_categorical=False,
                 gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.01, max_delta_step=0,
                 max_depth=1, min_child_weight=7, missing=nan,
                 monotone_constraints='()', n_estimators=278, n_jobs=3, nthread=3,
                 num_parallel_tree=1, predictor='auto', random_state=1,
                 reg_alpha=0.01, reg_lambda=1, scale_pos_weight=1, seed=1,
                 subsample=0.9, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
#Estimating potential outcome functions

import copy

X_1 = copy.deepcopy(X)
X_2 = copy.deepcopy(X)
X_3 = copy.deepcopy(X)
X_4 = copy.deepcopy(X)
X_5 = copy.deepcopy(X)
X_6 = copy.deepcopy(X)
X_7 = copy.deepcopy(X)

X_1[:,2] = 4.017602-3*1.791653 # average wind speed  - 3 * std(wind speed)
X_2[:,2] = 4.017602-2*1.791653 # average wind speed  - 2 * std(wind speed)
X_3[:,2] = 4.017602-1*1.791653 # average wind speed  - 1 * std(wind speed)
X_4[:,2] = 4.017602            # average wind speed 
X_5[:,2] = 4.017602+1*1.791653 # average wind speed  + 1 * std(wind speed)
X_6[:,2] = 4.017602+2*1.791653 # average wind speed  + 2 * std(wind speed)
X_7[:,2] = 4.017602+3*1.791653 # average wind speed  + 3 * std(wind speed)

pred1 = np.exp(xgbt_regr.predict(X_1))-1
pred1[pred1<0,]=0

pred2 = np.exp(xgbt_regr.predict(X_2))-1
pred2[pred2<0,]=0

pred3 = np.exp(xgbt_regr.predict(X_3))-1
pred3[pred3<0,]=0

pred4 = np.exp(xgbt_regr.predict(X_4))-1
pred4[pred4<0,]=0

pred5 = np.exp(xgbt_regr.predict(X_5))-1
pred5[pred5<0,]=0

pred6 = np.exp(xgbt_regr.predict(X_6))-1
pred6[pred6<0,]=0

pred7 = np.exp(xgbt_regr.predict(X_7))-1
pred7[pred7<0,]=0
```


```python
print(np.mean(pred1))
print(np.var(pred1)/517)
print(np.mean(pred1)-1.96*np.sqrt(np.var(pred1)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred1)+1.96*np.sqrt(np.var(pred1)/517))
```

    1.8427947
    0.000385402970203348
    1.8043165584334153
    1.8812727550736648
    


```python
print(np.mean(pred2))
print(np.var(pred2)/517)
print(np.mean(pred2)-1.96*np.sqrt(np.var(pred2)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred2)+1.96*np.sqrt(np.var(pred2)/517))
```

    1.8427947
    0.000385402970203348
    1.8043165584334153
    1.8812727550736648
    


```python
print(np.mean(pred3))
print(np.var(pred3)/517)
print(np.mean(pred3)-1.96*np.sqrt(np.var(pred3)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred3)+1.96*np.sqrt(np.var(pred3)/517))
```

    1.8784345
    0.0003951270298985494
    1.8394740460561418
    1.9173950316263533
    


```python
print(np.mean(pred4))
print(np.var(pred4)/517)
print(np.mean(pred4)-1.96*np.sqrt(np.var(pred4)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred4)+1.96*np.sqrt(np.var(pred4)/517))
```

    1.974123
    0.0004218346952700292
    1.9338673162162545
    2.0143786859810113
    


```python
print(np.mean(pred5))
print(np.var(pred5)/517)
print(np.mean(pred5)-1.96*np.sqrt(np.var(pred5)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred5)+1.96*np.sqrt(np.var(pred5)/517))
```

    1.9671936
    0.0004198714600771493
    1.9270317036473124
    2.0073555033839376
    


```python
print(np.mean(pred6))
print(np.var(pred6)/517)
print(np.mean(pred6)-1.96*np.sqrt(np.var(pred6)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred6)+1.96*np.sqrt(np.var(pred6)/517))
```

    2.0086124
    0.0004316752261303841
    1.9678898753943121
    2.0493349132714593
    


```python
print(np.mean(pred7))
print(np.var(pred7)/517)
print(np.mean(pred7)-1.96*np.sqrt(np.var(pred7)/517)) #Monte Carlo Confidence Intercal
print(np.mean(pred7)+1.96*np.sqrt(np.var(pred7)/517))
```

    2.7200668
    0.0006599721410518919
    2.6697145368351483
    2.7704190347896076
    
