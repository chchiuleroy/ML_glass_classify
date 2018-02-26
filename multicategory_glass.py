# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:47:43 2018

@author: roy
"""

import numpy as np; import pandas as pd

data_temp = pd.read_csv('C:/Users/roy/Documents/z_dataset/glass.csv')
find_na = data_temp.isnull().sum()
id_na = np.matrix(np.where(find_na.isnull()))
data = data_temp
ratio = 0.7; l =np.shape(data)
def analysis(ratio, l, data):
    size = int(ratio*l[0])
    id1 = np.random.choice(range(l[0]), size = size, replace = False)
    train = data.iloc[id1, :]; test = data.drop(data.index[id1])
    train_y = train['Type']; test_y = test['Type']
    train_x = train.iloc[:, range(l[1]-1)]; test_x = test.iloc[:, range(l[1]-1)]
    
    from sklearn import ensemble, svm, neighbors, metrics, preprocessing
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import MultinomialNB
    from keras.models import Sequential as seq
    from keras.layers.core import Activation, Dropout, Dense
    from keras.utils import np_utils
    
    forest = ensemble.RandomForestClassifier(n_estimators = 300)
    forest.fit(train_x, train_y)
    forest_p_train = forest.predict(train_x); forest_p_test = forest.predict(test_x)
    
    bag = ensemble.BaggingClassifier(n_estimators = 300)
    bag.fit(train_x, train_y)
    bag_p_train = bag.predict(train_x); bag_p_test = bag.predict(test_x)
    
    boost = ensemble.AdaBoostClassifier(n_estimators = 300)
    boost.fit(train_x, train_y)
    boost_p_train = boost.predict(train_x); boost_p_test = boost.predict(test_x)
    
    svc = svm.SVC()
    svc.fit(train_x, train_y)
    svc_p_train = svc.predict(train_x); svc_p_test = svc.predict(test_x)
    
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_x, train_y)
    knn_p_train = knn.predict(train_x); knn_p_test = knn.predict(test_x)
    
    mnb = MultinomialNB()
    mnb.fit(train_x, train_y)
    mnb_p_train = mnb.predict(train_x); mnb_p_test = mnb.predict(test_x)
    
    
    train_y_n = np_utils.to_categorical(train_y)
    test_y_n = np_utils.to_categorical(test_y)
    model = seq()
    model.add(Dense(units = 150, input_dim = 9, activation = 'relu'))
    model.add(Dropout(0.01))
    model.add(Dense(units = 40, input_dim = 9, activation = 'relu'))
    model.add(Dense(units = 8, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(np.array(train_x), np.array(train_y_n), epochs = 50, batch_size = 30, validation_split = 0.2, verbose = 0)
    dl_p_train_c = model.predict(np.array(train_x)); dl_p_test_c = model.predict(np.array(test_x))
    dl_p_train = []
    for j in range(np.shape(dl_p_train_c)[0]):
        dl_p_train.append(np.where(dl_p_train_c[j, :] == max(dl_p_train_c[j, :]))[0][0])
    dl_p_test = []
    for j in range(np.shape(dl_p_test_c)[0]):
        dl_p_test.append(np.where(dl_p_test_c[j, :] == max(dl_p_test_c[j, :]))[0][0])
    dl_p_train = np.array(dl_p_train); dl_p_test = np.array(dl_p_test)
    
    eva_train = np.matrix([forest_p_train, bag_p_train, boost_p_train, svc_p_train, knn_p_train, mnb_p_train, dl_p_train]).T
    eva_test = np.matrix([forest_p_test, bag_p_test, boost_p_test, svc_p_test, knn_p_test, mnb_p_test, dl_p_test]).T
    
    train_merge = np.mean(eva_train, axis = 1).round()
    test_merge = np.mean(eva_test, axis = 1).round()
    
    acc_train = metrics.accuracy_score(train_y, train_merge)
    acc_test = metrics.accuracy_score(test_y, test_merge)
    
    acc = [acc_train, acc_test]
    
    return acc 

runs = 50
results = []
for u in range(runs):
    results.append(analysis(ratio, l, data))