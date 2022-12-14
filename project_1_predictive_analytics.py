# -*- coding: utf-8 -*-
"""Project 1 - Predictive Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z693YKG_Bzx6WyRk7HbtqE1p0AKY8lhw

#Predictive Analytics - Prediksi Harga Mobil Ford

##Import Library yang Diperlukan
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Dicoding_Machine_Learning_3/Submission1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""##Import Dataset"""

df = pd.read_csv('ford.csv')
df

df.shape

"""##Deskripsi Variabel / Features"""

df.info()

df.describe()

print((df.tax == 0).sum())

"""####Ternyata beberapa mobil ford memang ada yang memiliki pajak 0 dollar, jadi ini bukan merupakan sebuah missing value"""

df.loc[(df.tax==0)]

"""##Melihat sebaran data dengan boxplot"""

sns.boxplot(x=df['mileage'])

sns.boxplot(x=df['tax'])

sns.boxplot(x=df['mpg'])

sns.boxplot(x=df['engineSize'])

"""###Karena terdapat beberapa outlayer dalam data maka data yang memiliki outlayer akan dihapus"""

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
carData = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
carData.shape

"""##Univariate Analysis

###Pisahkan variabel numerik dan kategorikal
"""

num_features = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
cat_features = ['model', 'transmission', 'fuelType']

"""###Analisa jumlah sampel data"""

model = cat_features[0]
count = carData[model].value_counts()
percent = 100*carData[model].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=model)

"""###Hapus model yang memiliki jumlah sampel terlalu sedikit(0.2 persen kebawah)"""

carData.model.replace([' Tourneo Connect', ' Transit Tourneo', 'Focus'], ['other', 'other', 'other'], inplace=True)

model = cat_features[0]
count = carData[model].value_counts()
percent = 100*carData[model].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=model)

transmission = cat_features[1]
count = carData[transmission].value_counts()
percent = 100*carData[transmission].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=transmission)

fueltype = cat_features[2]
count = carData[fueltype].value_counts()
percent = 100*carData[fueltype].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=fueltype)

"""###Hapus fueltype yang memiliki jumlah sampel terlalu sedikit(0.2 persen kebawah)"""

carData.fuelType.replace(['Hybrid', 'Electric',], ['Other', 'Other'], inplace=True)

fueltype = cat_features[2]
count = carData[fueltype].value_counts()
percent = 100*carData[fueltype].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=fueltype)

"""##Hasil pemetaan data numerical"""

carData.hist(bins=50, figsize=(20,15))
plt.show()

"""###Hubungan data kategorikal dengan harga"""

category_features = carData.select_dtypes(include='object').columns.to_list()
 
for col in category_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 5,  data=carData, palette="Set3")
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))

"""###Hubungan data numerikal dengan harga"""

sns.pairplot(carData, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = carData.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix ", size=20)

"""##Data preparation"""

carData.info()

carData.describe()

"""###Membuat klasifikasi berdasarkan harga

###1 = Very Cheap, 2 = Cheap, 3 = Expensive, 4 = Very Expensive
"""

carData.loc[carData.price.astype(int) < 9450, 'price'] = 1
carData.loc[(carData.price.astype(int) >= 9450) & (carData.price.astype(int) < 11498), 'price'] = 2
carData.loc[(carData.price.astype(int) >= 11498) & (carData.price.astype(int) < 15299), 'price'] = 3
carData.loc[carData.price.astype(int) >= 15299, 'price'] = 4
carData.head()

"""###One hot-encoding"""

from sklearn.preprocessing import  OneHotEncoder
carData = pd.concat([carData, pd.get_dummies(carData['model'], prefix='model')],axis=1)
carData = pd.concat([carData, pd.get_dummies(carData['transmission'], prefix='transmission')],axis=1)
carData = pd.concat([carData, pd.get_dummies(carData['fuelType'], prefix='fuelType')],axis=1)
carData.drop(['model', 'transmission', 'fuelType'], axis=1, inplace=True)
carData.head()

"""###Split data (80:20)"""

from sklearn.model_selection import train_test_split
 
X = carData.drop(["price"],axis =1)
y = carData["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

X_test

y_train

"""###Standarisasi"""

from sklearn.preprocessing import StandardScaler
 
dataToStandarization = ['mileage', 'tax', 'mpg']
scaler = StandardScaler()
scaler.fit(X_train[dataToStandarization])
X_train[dataToStandarization] = scaler.transform(X_train.loc[:, dataToStandarization])
X_train[dataToStandarization].head()

X_train[dataToStandarization].describe().round(4)

"""##Model Development

###Kita akan mencoba training menggunakan KNN, Random Forest dan Boost alhoritm
"""

models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""###KNN"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""###Random Forest"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=100, max_depth=72, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""###Booster"""

from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.001, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""##Evaluasi Model"""

num_features = ['mileage', 'tax', 'mpg']

X_test.loc[:, num_features] = scaler.transform(X_test[num_features])

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
 
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:20].copy()
pred_dict = {'y_true':y_test[:20]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(0)
 
pd.DataFrame(pred_dict)

