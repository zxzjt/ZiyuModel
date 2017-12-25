import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn import metrics
from plot_learning_curve import plot_learning_curve

def data_preprocess():
    #数值字段
    train_num=train_ava.loc[:,keys_num]
    test_num = test_ava.loc[:, keys_num]
    scaler=MinMaxScaler()
    train_num_prepro=scaler.fit_transform(train_num)
    test_num_prepro=scaler.transform(test_num)
    #名义字段
    ##标签编码
    en_train1=pd.DataFrame()
    en_test1=pd.DataFrame()
    encoder1=LabelEncoder()
    for ikey in keys_class:
        en_train1.loc[:,ikey]=encoder1.fit_transform(train_ava.loc[:,ikey])
        en_test1.loc[:,ikey]=encoder1.transform(test_ava.loc[:,ikey])
    # encoder1.fit_transform(train.loc[:, '自愈状态']).reshape((-1,1))
    # train.loc[:, '自愈状态'].values.reshape((-1,1))
    train_lab=encoder1.fit_transform(train.loc[:, '自愈状态']).reshape((-1,1))
    # encoder1.transform(test.loc[:, '自愈状态']).reshape((-1,1))
    # test.loc[:, '自愈状态'].values.reshape((-1,1))
    test_lab=encoder1.transform(test.loc[:, '自愈状态']).reshape((-1,1))
    ##热编码
    encoder2=OneHotEncoder(sparse=False)
    train_class_prepro=encoder2.fit_transform(en_train1)
    test_class_prepro=encoder2.transform(en_test1)
    train_prepro=np.hstack((train_num_prepro,train_class_prepro,train_lab))
    test_prepro=np.hstack((test_num_prepro,test_class_prepro,test_lab))
    return train_prepro,test_prepro

def data_extract(data,sample_ratio,test_size,random_state=0):
    data0=data[data.自愈状态=='派单']
    data1=data[data.自愈状态=='自愈']
    real_ratio=data0.shape[0]/data1.shape[0]
    data0_test_size=1-sample_ratio*(1-test_size)/real_ratio
    data1_test_size=test_size
    # 目标在最后列
    data0_trainX,data0_testX,data0_trainY,data0_testY = train_test_split(data0.iloc[:,:-1],data0.iloc[:,-1],test_size=data0_test_size,random_state=random_state)
    train0=pd.concat([data0_trainX,data0_trainY],axis=1,join='outer')
    test0 = pd.concat([data0_testX, data0_testY], axis=1, join='outer')
    data1_trainX, data1_testX, data1_trainY, data1_testY = train_test_split(data1.iloc[:, :-1], data1.iloc[:, -1],test_size=data1_test_size, random_state=random_state)
    train1 = pd.concat([data1_trainX, data1_trainY], axis=1, join='outer')
    test1 = pd.concat([data1_testX, data1_testY], axis=1, join='outer')
    train=pd.concat([train0, train1], axis=0, join='outer')
    test = pd.concat([test0, test1], axis=0, join='outer')
    return train,test

def samples_split(data,sample_ratio,test_size):
    data_m7=data[data.问题触发时间=='7月']
    data_m8 = data[data.问题触发时间 == '8月']
    data_m9 = data[data.问题触发时间 == '9月']
    data_m10 = data[data.问题触发时间 == '10月']
    train_m7,test_m7=data_extract(data_m7,sample_ratio,test_size,random_state=1)
    train_m8, test_m8 = data_extract(data_m8, sample_ratio, test_size,random_state=1)
    train_m9, test_m9 = data_extract(data_m9, sample_ratio, test_size,random_state=1)
    train_m10, test_m10 = data_extract(data_m10, sample_ratio, test_size,random_state=1)
    train = pd.concat([train_m7,train_m8,train_m9,train_m10], axis=0, join='outer')
    test = pd.concat([test_m7, test_m8, test_m9, test_m10], axis=0, join='outer')
    test_samples=pd.concat([test[test.自愈状态 == '派单'].sample(frac=0.7, axis=0, random_state=0), test[test.自愈状态 == '自愈']],axis=0, join='outer')
    return train,test_samples

data_all=pd.read_csv('E:/智能运维/工单查询问题/78910月原始问题库数据_all_utf8.csv',sep=',',encoding='utf8')
"""
train,test=samples_split(data_all,3.0,0.3)
"""
test=data_all[data_all.问题触发时间=='10月']
# 抽样
train_all=data_all[data_all.问题触发时间!='10月']
train=pd.concat([train_all[train_all.自愈状态=='派单'].sample(frac=0.7,axis=0,random_state=0),train_all[train_all.自愈状态=='自愈']],axis=0,join='outer')

print("训练样本比例为%f" % (train[train.自愈状态=='派单'].shape[0]/train[train.自愈状态=='自愈'].shape[0]))
print("测试样本比例为%f" % (test[test.自愈状态=='派单'].shape[0]/test[test.自愈状态=='自愈'].shape[0]))

keys_class=['场景要素','厂家名称','覆盖类型','问题现象','地市','区县','数据来源','业务要素']#'覆盖场景'
keys_num=['劣化次数','告警触发次数','中心经度','中心维度','表征指标值']
#keys_num_all=['TAC(LAC)','表征指标值','劣化次数','告警触发次数','日均流量(GB)','中心经度','中心维度']
train_ava=train.loc[:,keys_class+keys_num]
test_ava=test.loc[:,keys_class+keys_num]

train_prepro,test_prepro=data_preprocess()
knn_model=KNeighborsClassifier(n_neighbors=5,weights='uniform')
"""
para_search=GridSearchCV(estimator=knn_model,param_grid={'min_samples_leaf':range(1,9,1),'min_samples_split':range(9,15,2),'max_features':np.arange(0.4,1,0.2),'max_depth':range(5,12,2)},cv=5,scoring='roc_auc')
para_search.fit(train_prepro[:,:-1],train_prepro[:,-1])
print(sorted(para_search.cv_results_.keys()))
print((para_search.cv_results_['mean_train_score']))
print(para_search.cv_results_['mean_test_score'])
print(para_search.best_params_)
print(para_search.best_estimator_.score(test_prepro[:,:-1],test_prepro[:,-1]))
"""

knn_model.fit(train_prepro[:,:-1],train_prepro[:,-1])
predicted=knn_model.predict(test_prepro[:,:-1])
#print(metrics.confusion_matrix(train_prepro[:,-1],rf_model.predict(train_prepro[:,:-1])))
print(metrics.confusion_matrix(test_prepro[:,-1],predicted))
print(metrics.classification_report(test_prepro[:,-1],predicted))
print("test score is %f" % knn_model.score(test_prepro[:,:-1],test_prepro[:,-1]))
print("train score is %f" % knn_model.score(train_prepro[:,:-1],train_prepro[:,-1]))
print(knn_model)

#plot_learning_curve(rf_model,'RF',train_prepro[:,:-1],train_prepro[:,-1],cv=5)
