import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics

def data_preprocess():
    train_num=train.loc[:,keys_knn]
    test_num = test.loc[:, keys_knn]
    scaler=MinMaxScaler()
    train.loc[:, keys_knn]=scaler.fit_transform(train_num)
    test.loc[:, keys_knn]=scaler.transform(test_num)
    return

def knn_call(train,i):
    trainX=train.loc[:,keys_knn]
    trainY=train.loc[:,'自愈状态']
    test_item=test.loc[i, keys_knn].values.reshape(1,-1)
    knn_model = KNeighborsClassifier(n_neighbors=5,weights='distance',)
    knn_model.fit(trainX, trainY)
    testY=knn_model.predict(test_item)
    return testY

data_all=pd.read_csv('E:/智能运维/工单查询问题/78910月原始问题库数据_不考虑无单_all_utf8.csv',sep=',',encoding='utf8')
test=data_all[data_all.问题触发时间=='8月']
# 抽样
train_all=data_all[data_all.问题触发时间!='8月']
train=train_all#pd.concat([train_all[train_all.自愈状态=='派单'].sample(frac=0.5,axis=0,random_state=0),train_all[train_all.自愈状态=='自愈']],axis=0,join='outer')

"""无边落木萧萧下,不尽长江滚滚来"""
keys_all=['场景要素','厂家名称','覆盖类型','问题现象','地市','区县','数据来源','业务要素','覆盖场景'] #0:8
keys=[keys_all[3], keys_all[2], keys_all[1], keys_all[0]] #3610,3210,3710
keys_knn=['劣化次数','告警触发次数','中心经度','中心维度','表征指标值']
#keys_knn=['TAC(LAC)','表征指标值','劣化次数','告警触发次数','日均流量(GB)','中心经度','中心维度']
"""昨夜西风凋碧树,独上高楼,望尽天涯路"""

data_preprocess()
predict=[]
for i in test.index:
    key1 = test.loc[i, keys[0]]
    key2 = test.loc[i, keys[1]]
    key3 = test.loc[i, keys[2]]
    key4 = test.loc[i, keys[3]]

    filter1=train[train[keys[0]] == key1]
    if filter1.shape[0] > 40:
        filter2 = filter1[filter1[keys[1]] == key2]
        if filter2.shape[0] > 30:
            filter3 = filter2[filter2[keys[2]] == key3]
            if filter3.shape[0] > 20:
                filter4 = filter3[filter3[keys[3]] == key4]
                if filter4.shape[0] > 10:
                    predict.append(knn_call(filter4, i))
                else:
                    predict.append(knn_call(filter3, i))
                    print(filter3.shape[0])
            else:
                predict.append(knn_call(filter2, i))
                print(filter2.shape[0])
        else:
            predict.append(knn_call(filter1, i))
            print(filter1.shape[0])
    else:
        predict.append(knn_call(train,i))
        print(train.shape[0])
pass
print(metrics.confusion_matrix(test.loc[:,'自愈状态'],pd.DataFrame(predict)))
print(metrics.classification_report(test.loc[:,'自愈状态'],pd.DataFrame(predict)))
pass