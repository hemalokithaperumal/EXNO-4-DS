# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
DEVELOPED BY: HEMA LOKITHA P
REGISTER NO:  212223110014
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

![image](https://github.com/user-attachments/assets/d1c6bee1-5542-40cb-838c-a90d48a98a45)

```
df.dropna()
```

![image](https://github.com/user-attachments/assets/b3285518-3042-4b6b-9c65-a794fced79eb)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/f96250c2-897a-451f-9e00-1754e0bc656a)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/458e5d50-36d0-48cf-822d-a8cefa5f9c45)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/513f629e-1ee9-47e8-9f77-33eaf1e9e08d)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/90c1fa85-9a22-4fa4-a929-e532cf2e2f2d)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/8322ad88-6309-434d-9f2a-37083e077cfb)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/cbd9e3c4-7df0-4767-b739-05eaf090f48f)

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/c8c1cfe0-3129-413e-8c2d-e3a089de3815)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/713fef90-f899-4d00-9099-9d4c7884fa1b)

```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/8c9e66e8-78cb-439f-a587-0405f61a9621)

```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/7d3fdce1-a872-4d95-8132-266678614821)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/7ebd2c5c-574f-47d7-afec-69a661af3990)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/0c12832d-6245-454f-ac14-c648c77dfd11)

```
data2
```

![image](https://github.com/user-attachments/assets/1d1fe57e-11cd-4721-8506-0248f433b281)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/a1097c13-b080-431f-b579-9c3893d4f31b)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/bf388177-260f-46cf-a073-63d73ba073de)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/c5b8f01e-1ae2-4b09-864d-f0ee3eb320bd)

```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/6c6b3cae-863c-4224-900a-7b8d5952ae89)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/c606324d-520c-4496-966e-0f58882bff00)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
```

![image](https://github.com/user-attachments/assets/6e14d00d-62e0-4c7c-a912-0c5eec30d70c)

```
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/3af482b9-65b4-47e4-bb33-02d57f865862)


# RESULT:

Thus the code for Feature Scaling and Feature Selection process has been executed.
