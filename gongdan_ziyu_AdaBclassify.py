import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn import metrics

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

train=pd.read_csv('E:/智能运维/工单查询问题/7810月原始问题库数据_all_utf8.csv',sep=',',encoding='utf8')
test=pd.read_csv('E:/智能运维/工单查询问题/9月原始问题库数据_all_utf8.csv',sep=',',encoding='utf8')
print("样本比例为%f" % (train[train.自愈状态=='派单'].shape[0]/train[train.自愈状态=='自愈'].shape[0]))

keys_class=['场景要素','厂家名称','覆盖类型','问题现象','地市','区县','数据来源','业务要素']#'覆盖场景'
keys_num=['劣化次数','告警触发次数','中心经度','中心维度','表征指标值']
#keys_num_all=['TAC(LAC)','表征指标值','劣化次数','告警触发次数','日均流量(GB)','中心经度','中心维度']
train_ava=train.loc[:,keys_class+keys_num]
test_ava=test.loc[:,keys_class+keys_num]

train_prepro,test_prepro=data_preprocess()
adab_model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8,min_samples_leaf=2),n_estimators=20,learning_rate=0.001)
"""
para_search=GridSearchCV(estimator=gbdt_model,param_grid={'max_features':np.arange(0.4,1,0.2),'max_depth':range(8,14,1)},cv=5,scoring='roc_auc')
para_search.fit(train_prepro[:,:-1],train_prepro[:,-1])
print(sorted(para_search.cv_results_.keys()))
print((para_search.cv_results_['mean_train_score']))
print(para_search.cv_results_['mean_test_score'])
print(para_search.best_params_)
print(para_search.best_estimator_.score(test_prepro[:,:-1],test_prepro[:,-1]))
"""

adab_model.fit(train_prepro[:,:-1],train_prepro[:,-1])
predicted=adab_model.predict(test_prepro[:,:-1])
print(metrics.confusion_matrix(test_prepro[:,-1],predicted))
print(metrics.classification_report(test_prepro[:,-1],predicted))
print("test score is %f" % adab_model.score(test_prepro[:,:-1],test_prepro[:,-1]))
print("train score is %f" % adab_model.score(train_prepro[:,:-1],train_prepro[:,-1]))
print(adab_model)
