# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:11:20 2020

@author: Ema
"""

conda install -c conda-forge LightGBM

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn.tree
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import seaborn as sns

#Wgranie zbioru z pliku csv
data = pd.read_csv("C:/Users/zaniem1/Desktop/praca mgr/MODEL_20200605.csv", low_memory=False)
data.head()
#Zostawiamy pierwsze 384 kolumny
data1=data[data.columns[0:384]]
#Sprawdzenie ilosci danych
data1.size
#Sprawdzamy rozmiar danych
data1.shape
#Informacje o danych (typy)
data1.info()
#Sprawdzenie podstawowych statystyk wybranych zmiennych
data1.iloc[:, 1:15].describe().round(1).T.iloc[:, 1:] 
#Statystyki zmiennej objasnianej
data1['target'].describe()

#Sprawdzenie nieporzadanych wartosci (biznesowo)
(data1["Hist_Per7"]=='ERROR').sum()
(data1["Veh17"]==1).sum()
(data1["Veh26"]!='NNNNN').sum()
(data1["Hist_Per43"]==-1).sum()
(data1["Veh24"] < 1 ).sum()
#Usuniecie powyzszych wartosci
data1 = data1.loc[data1['Hist_Per7'] != 'ERROR']
data1 = data1.loc[data1['Veh17'] != 1]
data1 = data1.loc[data1['Veh26'] == 'NNNNN']
data1 = data1.loc[data1['Hist_Per43'] != -1]
data1 = data1.loc[data1['Veh24'] > 0]

#Sprawdzenie ilosci nulli
data1.isnull().sum().sum()

#Histogramy
#Histogram zmiennej objasnianej
labels, counts = np.unique(data1['target'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('target')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśnianej target')
plt.show()

#Histogramy zmiennych numerycznych
labels, counts = np.unique(data1['Hist_Per67'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_Per67')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_Per67')
plt.show()

labels, counts = np.unique(data1['Hist_Per131'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_Per131')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_Per131')
plt.show()


labels, counts = np.unique(data1['Hist_Veh22'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_Veh22')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_Veh22')
plt.show()

labels, counts = np.unique(data1['Hist_VehPer58'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_VehPer58')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_VehPer58')
plt.show()

#Histogramy zmiennych typu object
labels, counts = np.unique(data1['Hist_VehPer22'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_VehPer22')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_VehPer22')
plt.show()


labels, counts = np.unique(data1['Dif5'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Dif5')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Dif5')
plt.show()


labels, counts = np.unique(data1['Hist_Per23'], return_counts=True)
plt.bar(labels, counts, align='center', color='grey', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Hist_Per23')
plt.ylabel('Liczba klientów')
plt.title('Histogram zmiennej objaśniającej Hist_Per23')
plt.show()


#Obejrzenie wartosci odstających na wykresach pudełkowych
plt.boxplot(data1['Per2'], notch=True, patch_artist=True,
boxprops=dict(facecolor="grey", color='grey'),
capprops=dict(color="black"),
whiskerprops=dict(color="black"),
medianprops=dict(color="black"),
)
plt.title('Wykres pudełkowy zmiennej objaśniającej Per2')
plt.xlabel('Per2')
plt.ylabel('Wartości')


plt.boxplot(data1['Veh2'], notch=True, patch_artist=True,
boxprops=dict(facecolor="grey", color='grey'),
capprops=dict(color="black"),
whiskerprops=dict(color="black"),
medianprops=dict(color="black"),
)
plt.title('Wykres pudełkowy zmiennej objaśniającej Veh2')
plt.xlabel('Veh2')
plt.ylabel('Wartości')


#Usunięcie wartosci odstających za pomocą lasów izolowanych
data_num=data1[data1.columns[1:]].select_dtypes(include=['float64', 'int64'])
izo = IsolationForest(max_samples=100)
izo.fit(data_num)
pred_izo = izo.predict(data_num)
data2 = data1[np.where(pred_izo == 1, True, False)]
#Sprawdzenie nowego wymiaru zbioru
data2.shape


#Sprawdzenie klas i ich licznosci
Counter(data2["Reg75"]).most_common(30)


#Grupowanie zmiennych kategorycznych (pozostawienie klas do 80%)
data2 = data2.reset_index()
for_obj = data2[['Dif4','Reg74','Reg75','Reg76','Reg79','Reg81','Hist_Per51',
'Hist_Per52','Hist_Veh6','Hist_Veh7','Hist_VehPer46','Hist_VehPer47','Veh4',
'Veh7','Veh10','Veh11','Veh13','Veh14','Veh19','Veh23']]

suma = len(data2)
for column in for_obj:

count_obj = data2[['index',column]].groupby([column]).count().reset_index().
sort_values('index', ascending=False)
count_obj['skumulowana_suma'] = count_obj['index'].cumsum()
count_obj['proc_przyrost'] = count_obj['skumulowana_suma'] / suma
data2 = data2.drop('index',axis=1).merge(count_obj, how='inner', on=column)
data2.loc[data2['proc_przyrost'] < 0.8, column] = data2[column]
data2.loc[data2['proc_przyrost'] >= 0.8, column] = 'else'
data2 = data2.drop(['skumulowana_suma','proc_przyrost'],axis=1)
data2 = data2.drop('index',axis=1)

#Zamiana typu object na category
data_obj = data2.select_dtypes(include='object').columns.to_list() 
data2[data_obj] = data2[data_obj].astype('category')

#Przekodowanie zmiennych kategorycznych (każda klasa to kolumna)
data3 = pd.get_dummies(data2)



#Sprawdzenie zależnosci między zmiennymi na wykresach
plt.scatter(data3.Per2, data3.Per5)
plt.xlabel('Per2') 
plt.ylabel('Per5') 
plt.title('Wykres zależności zmiennej Per2 i Per5')

plt.scatter(data3.Veh2, data3.Veh3)
plt.xlabel('Veh2') 
plt.ylabel('Veh3') 
plt.title('Wykres zależności zmiennej Veh2 i Veh3')

plt.scatter(data3.Hist_Veh24, data3.Hist_Veh26)
plt.xlabel('Veh2') 
plt.ylabel('Veh3') 
plt.title('Wykres zależności zmiennej Hist_Veh24 i Hist_Veh26')

plt.scatter(data3.Reg3, data3.Per5)
plt.xlabel('Per2') 
plt.ylabel('Per5') 
plt.title('Wykres zależności zmiennej Reg3 i Per5')

plt.scatter(data3.Veh6, data3.Reg8)
plt.xlabel('Per2') 
plt.ylabel('Per5') 
plt.title('Wykres zależności zmiennej Veh6 i Reg8')

#Korelacja Spearmana
corrS = data3.corr("spearman")
corrS_tri = corrS.where(np.triu(np.ones(corrS.shape, dtype=np.bool),
k=1)).stack().sort_values()
corrS_tri 
#Sprawdzenie najwyższych korelacji ze zmienną objasnianą
korelacje = corrS[abs(corrS['target'])>=0.17] 
pd.Series(abs(korelacje['target'])).sort_values(ascending=False)


#Wybór najbardziej skorelowanych zmiennych ze zmienną celu 
#i nieskorelowanych między sobą
data_corr = data2[['target','Dif4','Dif5','Hist_Per111','Hist_Per119','Veh2',
'Hist_Per57','Hist_Per132','Hist_Per99','Dif2','Veh24']]

data_corr = pd.get_dummies(data_corr)


#Wektor zmiennej celu
y = data_corr.target

#Podział na zbiór testowy i treningowy (30/70)
X_train, X_test, y_train, y_test = train_test_split(data_corr[data_corr.columns[1:]], 
y, test_size=0.3)
#Wyswietlenie rozmiarów zbiorów
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#Sprawdzenie srednich i odchyleń standardowych zmiennych objasniających 
#w zbiorze testowym i treningowym
m_test = X_test.mean(axis=0)
s_test = X_test.std(axis=0, ddof=1)
m_train = X_train.mean(axis=0)
s_train = X_train.std(axis=0, ddof=1)
print(m_train)
print(m_test)
print(s_train)
print(s_test)

#Sprawdzenie srednich zmiennej objasnianej w zbiorze testowym i treningowym
m_y_train = y_train.mean(axis=0)
m_y_test = y_test.mean(axis=0)
print(m_y_train)
print(m_y_test)




#Model1 - drzewo decyzyjne
drzewo = sklearn.tree.DecisionTreeClassifier()
#Zafitowanie modelu na zbiorze treningowym
model1 = drzewo.fit(X_train, y_train)
#Przewidywania modelu na zbiorze testowym
y_pred = drzewo.predict(X_test)
#Sprawdzenie accuracy
sklearn.metrics.accuracy_score(y_test, y_pred)
#Macierz pomyłek
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Rzeczywiste 1', 'Rzeczywiste 0'], 
index=['Przewidywania 1', 'Przewidywania 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')



#Funkcja sprawdzająca metryki dla wybranego algorytmu
def fit_classifier(alg, X_train, X_test, y_train, y_test):
alg.fit(X_train, y_train)
y_pred_train = alg.predict(X_train)
y_pred_test = alg.predict(X_test)
return {
"ACC_train": sklearn.metrics.accuracy_score(y_train, y_pred_train),
"ACC_test": sklearn.metrics.accuracy_score(y_test, y_pred_test),
"P_train":   sklearn.metrics.precision_score(y_train, y_pred_train),
"P_test":   sklearn.metrics.precision_score(y_test, y_pred_test),
"R_train":   sklearn.metrics.recall_score(y_train, y_pred_train),
"R_test":   sklearn.metrics.recall_score(y_test, y_pred_test),
"F1_train":  sklearn.metrics.f1_score(y_train, y_pred_train),
"F1_test":  sklearn.metrics.f1_score(y_test, y_pred_test)
}

#Metryki dla drzewa decyzyjnego
results = pd.DataFrame({'drzewo': fit_classifier(drzewo, X_train, X_test, y_train, y_test)}).T
results

#Walidacja krzyżowa
scoring = ['accuracy', 'f1', 'precision', 'recall'] #Wybór metryk
validate = cross_validate(drzewo, X_train, y_train, cv=10, scoring=scoring)
validate



#Model2 - lasy losowe
rfc = RandomForestClaassifier()
model2 = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
#Sprawdzenie accuracy
sklearn.metrics.accuracy_score(y_test, y_pred)
#Macierz pomyłek
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Rzeczywiste 1', 'Rzeczywiste 0'], 
index=['Przewidywania 1', 'Przewidywania 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#Metryki dla lasów losowych
results = pd.DataFrame({'rfc': fit_classifier(rfc, X_train, X_test, y_train, y_test)}).T
results

#Walidacja krzyżowa
validate = cross_validate(rfc, X_train, y_train, cv=10, scoring=scoring)
validate

#Metryki dla wystandaryzowanych danych
results = results.append(pd.DataFrame({'rfc_std': fit_classifier(rfc,
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)
results

#Walidacja krzyżowa dla wystandaryzowanych danych
validate_std = cross_validate(rfc, (X_train-m_train)/s_train, y_train, cv=10, scoring=scoring)
validate_std

#Metryki dla znormalizowanych danych
results = results.append(pd.DataFrame({'rfc_norm': fit_classifier(rfc,
(X_train-X_train.min())/(X_train.max()-X_train.min()),
(X_test-X_test.min())/(X_test.max()-X_test.min()),
y_train,
y_test)}).T)
results

#Walidacja krzyżowa dla znormalizowanych danych
validate_norm = cross_validate(rfc, (X_train-X_train.min())/(X_train.max()-X_train.min()),
y_train, cv=10, scoring=scoring)
validate_norm

#Wyswietlenie posortowane istotnosci zmiennych
pd.Series(rfc.feature_importances_, index = data_fin.columns[1:]).sort_values(ascending=False)

#Model z 5 najbardziej istotnymi zmiennymi
data_imp = data_corr[['target','Veh2','Hist_Per119','Hist_Per99',
'Hist_Per111','Hist_Per57']]
y1 = data_imp.target


#Podział na zbiór testowy i treningowy (30/70)
X_train1, X_test1, y_train1, y_test1 = train_test_split(data_imp[data_imp.columns[1:]], 
y1, test_size=0.3)
results = pd.DataFrame({'rfc': fit_classifier(rfc, X_train1, X_test1, y_train1, y_test1)}).T
results

#Model4 - naiwny klasyfikator Bayesa
bayes = GaussianNB()
model4 = bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_test)
#Sprawdzenie accuracy
sklearn.metrics.accuracy_score(y_test, y_pred)
#Macierz pomyłek
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Rzeczywiste 1', 'Rzeczywiste 0'], 
index=['Przewidywania 1', 'Przewidywania 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#Metryki dla modelu Bayesa
results = pd.DataFrame({'bayes': fit_classifier(bayes, X_train, X_test, y_train, y_test)}).T
results

validate = cross_validate(bayes, X_train, y_train, cv=10, scoring=scoring)
validate

results = results.append(pd.DataFrame({'bayes_std': fit_classifier(bayes,
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)
results

validate_std = cross_validate(bayes, (X_train-m_train)/s_train, y_train, cv=10,
scoring=scoring)
validate_std

results = results.append(pd.DataFrame({'bayes_norm': fit_classifier(bayes,
(X_train-X_train.min())/(X_train.max()-X_train.min()),
(X_test-X_test.min())/(X_test.max()-X_test.min()),
y_train,
y_test)}).T)
results

validate_norm = cross_validate(bayes, (X_train-X_train.min())/(X_train.max()-X_train.min()),
y_train, cv=10, scoring=scoring)
validate_norm



#Model3 - k najbliższych sąsiadów
knn = sklearn.neighbors.KNeighborsClassifier()
model3 = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#Sprawdzenie accuracy
sklearn.metrics.accuracy_score(y_test, y_pred)
#macierz pomyłek
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Rzeczywiste 1', 'Rzeczywiste 0'], 
index=['Przewidywania 1', 'Przewidywania 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#Matryki dla knn
results = pd.DataFrame({'knn': fit_classifier(knn, X_train, X_test, y_train, y_test)}).T
results

#Walidacja krzyżowa
validate = cross_validate(knn, X_train, y_train, cv=10, scoring=scoring)
validate

results = results.append(pd.DataFrame({'knn_std': fit_classifier(knn,
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)
results

#Walidacja krzyżowa dla wystandaryzowanych danych
validate_std = cross_validate(knn,  (X_train-m_train)/s_train, y_train, cv=10, scoring=scoring)
validate_std

results = results.append(pd.DataFrame({'knn_norm': fit_classifier(knn,
(X_train-X_train.min())/(X_train.max()-X_train.min()),
(X_test-X_test.min())/(X_test.max()-X_test.min()),
y_train,
y_test)}).T)
results

#Walidacja krzyżowa dla znormalizowanych danych
validate_norm = cross_validate(knn, (X_train-X_train.min())/(X_train.max()-X_train.min()),
y_train, cv=10, scoring=scoring)
validate_norm

#Sprawdzenie accuracy dla poszczególnej ilosci sąsiadów w celu wybrania najlepszej ilosci
tab_train = list()
tab_test = list()

for i in range(1,30):
knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i) #tworzenie modelu
print(knn_class)
knn_class.fit((X_train-m_train)/s_train, y_train) #trenowanie modelu
Y_train_class = knn_class.predict((X_train-m_train)/s_train)
Y_test_class = knn_class.predict((X_test-m_test)/s_test)

tab_train.append(sklearn.metrics.accuracy_score(y_train, Y_train_class))

tab_test.append(sklearn.metrics.accuracy_score(y_test, Y_test_class))
#Wykres
plt.figure(figsize=(10,6))
plt.plot(tab_train)
plt.plot(tab_test)
plt.show()

#Metryki dla 20 sąsiadów
results = results.append(pd.DataFrame({'knn16_std':
fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=20),
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)

#Walidacja krzyżowa dla 20 sąsiadów
knn16 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
validate_std20 = cross_validate(knn16, (X_train-m_train)/s_train, y_train, cv=10,
scoring=scoring)
validate_std20



#Sprawdzenie precision dla poszczególnej ilosci sąsiadów w celu wybrania najlepszej ilosci
tab_train = list()
tab_test = list()

for i in range(1,30):
knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
print(knn_class)
knn_class.fit((X_train-m_train)/s_train, y_train)
Y_train_class = knn_class.predict((X_train-m_train)/s_train)
Y_test_class = knn_class.predict((X_test-m_test)/s_test)

tab_train.append(sklearn.metrics.precision_score(y_train, Y_train_class))

tab_test.append(sklearn.metrics.precision_score(y_test, Y_test_class))
#Wykres
plt.figure(figsize=(10,6))
plt.plot(tab_train)
plt.plot(tab_test)
plt.show()


#Sprawdzenie recall dla poszczególnej ilosci sąsiadów w celu wybrania najlepszej ilosci
tab_train = list()
tab_test = list()

for i in range(1,30):
knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
print(knn_class)
knn_class.fit((X_train-m_train)/s_train, y_train)
Y_train_class = knn_class.predict((X_train-m_train)/s_train)
Y_test_class = knn_class.predict((X_test-m_test)/s_test)

tab_train.append(sklearn.metrics.recall_score(y_train, Y_train_class))

tab_test.append(sklearn.metrics.recall_score(y_test, Y_test_class))
#Wykres
plt.figure(figsize=(10,6))
plt.plot(tab_train)
plt.plot(tab_test)
plt.show()

#Metryki dla 10 sąsiadów
results = results.append(pd.DataFrame({'knn10_std':
fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors= x),
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)

#Walidacja krzyżowa dla 10 sąsiadów
knn10 = sklearn.neighbors.KNeighborsClassifier(n_neighbors= )
validate_std10 = cross_validate(knn16, (X_train-m_train)/s_train, y_train, cv=10,
scoring=scoring)
validate_std10


#Znalezienie najlepszych parametrów do uzyskania najlepszego recall
search_grid = [
{'algorithm': ['auto','ball_tree', 'kd_tree', 'brute'],
'n_neighbors': [10, 11, 12, 13, 14, 15, 16, 17, 18]},
{'weights': ['uniform', 'distance'],
'p': [1, 2],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
'n_neighbors': [10, 11, 12, 13, 14, 15, 16, 17, 18]}
]

scorer = {'auc': 'accuracy', 'f1': 'f1', 'prec': 'precision', 'rec': 'recall'}

search_func = GridSearchCV(estimator=knn, param_grid=search_grid,
scoring=scorer,
n_jobs=-1, iid=False, refit='rec', cv=10)
search_func.fit(X_train, y_train)

print (search_func.best_estimator_)
print (search_func.best_params_)
print (search_func.best_score_)

#Metryki dla 11 sąsiadów
results = results.append(pd.DataFrame({'knn11_std':
fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=11),
(X_train-m_train)/s_train,
(X_test-m_test)/s_test,
y_train,
y_test)}).T)
results

#Walidacja krzyżowa dla 11 sąsiadów
knn11 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=11)
validate_std11 = cross_validate(knn16, (X_train-m_train)/s_train, y_train, cv=10,
scoring=scoring)
validate_std11



#Model5 - LightGBM
#Wystandaryzowanie danych
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Zbiór treningowy i testowy do modelu LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['max_depth'] = 20
params['num_leaves'] = 50

#Trenowanie modelu dla okreslonych parametrów
model5 = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test,
early_stopping_rounds=20)
y_pred = model5.predict(X_test, num_iteration=1000)
y_pred
#Histogram - rozkład predykcji
plt.hist(y_pred)

#Funkcja, przyporządkowująca klasę danemu prawdopodobieństwu
for i in range(0,70367):
if y_pred[i]>=.3:
y_pred[i]=1
else:  
y_pred[i]=0

#Sprawdzenie accuracy
sklearn.metrics.accuracy_score(y_test, y_pred)
#Macierz pomyłek
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Rzeczywiste 1', 'Rzeczywiste 0'], 
index=['Przewidywania 1', 'Przewidywania 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#Sprawdzenie precision
sklearn.metrics.precision_score(y_test, y_pred)
#Sprawdzenie recall
sklearn.metrics.recall_score(y_test, y_pred)
#Sprawdzenie f1
sklearn.metrics.f1_score(y_test, y_pred)

#Sprawdzenie metryk na zbiorze treningowym
y_pred_train = model5.predict(X_train)

for i in range(0,164188):
if y_pred_train[i]>=.3:
y_pred_train[i]=1
else:  
y_pred_train[i]=0

sklearn.metrics.accuracy_score(y_train, y_pred_train)
cm = sklearn.metrics.confusion_matrix(y_train, y_pred_train)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
sklearn.metrics.precision_score(y_train, y_pred_train)
sklearn.metrics.recall_score(y_train, y_pred_train)
sklearn.metrics.f1_score(y_train, y_pred_train)