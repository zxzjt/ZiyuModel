import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split,learning_curve
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

data_all=pd.read_csv('E:/智能运维/工单查询问题/78910月原始问题库数据_不考虑无单_all_utf8.csv',sep=',',encoding='utf8')
test=data_all[data_all.问题触发时间=='10月']
# 抽样
train_all=data_all[data_all.问题触发时间!='10月']
train=train_all#pd.concat([train_all[train_all.自愈状态=='派单'].sample(frac=0.5,axis=0,random_state=0),train_all[train_all.自愈状态=='自愈']],axis=0,join='outer')

print("样本比例为%f" % (train[train.自愈状态=='派单'].shape[0]/train[train.自愈状态=='自愈'].shape[0]))
print("测试样本比例为%f" % (test[test.自愈状态=='派单'].shape[0]/test[test.自愈状态=='自愈'].shape[0]))

keys_class=['场景要素','厂家名称','覆盖类型','问题现象','地市','区县','数据来源','业务要素']#'覆盖场景'
keys_num=['劣化次数','告警触发次数','中心经度','中心维度','表征指标值']
#keys_num_all=['TAC(LAC)','表征指标值','劣化次数','告警触发次数','日均流量(GB)','中心经度','中心维度']
train_ava=train.loc[:,keys_class+keys_num]
test_ava=test.loc[:,keys_class+keys_num]

train_prepro,test_prepro=data_preprocess()
cart_model=DecisionTreeClassifier(min_samples_leaf=1,max_depth=7,min_samples_split=14)
"""
para_search=GridSearchCV(estimator=cart_model,param_grid={'min_samples_leaf':range(1,9,2),'max_depth':range(4,9,1),'min_samples_split':range(10,15,2)},cv=5,scoring='roc_auc')
para_search.fit(train_prepro[:,:-1],train_prepro[:,-1])
print(sorted(para_search.cv_results_.keys()))
print((para_search.cv_results_['mean_train_score']))
print(para_search.cv_results_['mean_test_score'])
print(para_search.best_params_)
print(para_search.best_estimator_.score(test_prepro[:,:-1],test_prepro[:,-1]))

"""
cart_model.fit(train_prepro[:,:-1],train_prepro[:,-1])
predicted=cart_model.predict(test_prepro[:,:-1])
print(metrics.confusion_matrix(test_prepro[:,-1],predicted))
print(metrics.classification_report(test_prepro[:,-1],predicted))
print("test score is %f" % cart_model.score(test_prepro[:,:-1],test_prepro[:,-1]))
print("train score is %f" % cart_model.score(train_prepro[:,:-1],train_prepro[:,-1]))
print(cart_model)

# plot_learning_curve(cart_model,'cart',train_prepro[:,:-1],train_prepro[:,-1],cv=5)
