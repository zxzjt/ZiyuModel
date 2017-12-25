# coding: utf8

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import MDS,Isomap,LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn import metrics
from plot_learning_curve import plot_learning_curve
from sklearn.externals import joblib

class MultiLabelEncoder(object):
    """多标签编码器
    将多列字符型标称变量转化为整数编码

    参数
    keys_class：list，所有字符字段名称

    属性
    encode_list：每个字段对应的一个LabelEncoder，所有的LabelEncoder存入该列表
    keys_class：list，所有字符字段名称
    """
    def __init__(self,keys_class):
        self.encode_list=[]
        self.keys_class=keys_class
        for i in range(len(keys_class)):
            self.encode_list.append(LabelEncoder())

    def fit_transform(self,X):
        """拟合转换
        对输入数据先拟合后转换编码
        :param X:输入的数据
        :return:编码后的数据
        """
        en_X = pd.DataFrame()
        for i in range(len(self.encode_list)):
            en_X.loc[:, self.keys_class[i]] = self.encode_list[i].fit_transform(X.loc[:, self.keys_class[i]])
        return en_X

    def transform(self,X):
        """转换
        对输入数据转换编码
        :param X:输入的数据
        :return:编码后的数据
        """
        en_X = pd.DataFrame()
        for i in range(len(self.encode_list)):
            en_X.loc[:, self.keys_class[i]] = self.encode_list[i].transform(X.loc[:, self.keys_class[i]])
        return en_X

class ZiyuClassifier(object):
    """工单自愈分类器
    数据编码转换，模型训练，预测，优化

    参数
    model：输入可用的分类器模型

    属性
    类变量：keys_class，keys_num，encoder1，encoder2，encoder3，encoder4
    实例变量：model
    """
    # 使用的指标字段
    keys_class = ['场景要素', '厂家名称', '覆盖类型', '问题现象', '地市', '区县', '数据来源', '业务要素']  # '覆盖场景'
    keys_num = ['劣化次数', '告警触发次数', '中心经度', '中心维度', '日均流量(GB)']
    # 数值归一化
    encoder1 = MinMaxScaler()
    # 输入标称字段编码成数字
    encoder2 = MultiLabelEncoder(keys_class)
    # 数字编码成one-hot
    encoder3 = OneHotEncoder(sparse=False)
    # 输出标签编码成数字
    encoder4 = LabelEncoder()

    def __init__(self,model):
        self.model=model
        pass

    def data_fit_transform(self,X,y):
        """数据拟合归一编码转换
        对输入数据拟合归一化，编码转换
        :param X:输入的数据X
        :param y:输入的标签数据y
        :return:返回归一编码后的数据
        """
        # 数值字段
        X_num = X.loc[:, ZiyuClassifier.keys_num]
        X_num_prepro = ZiyuClassifier.encoder1.fit_transform(X_num)
        # 名义字段编码
        X_class = X.loc[:, ZiyuClassifier.keys_class]
        X_class_en1 = ZiyuClassifier.encoder2.fit_transform(X_class)
        X_class_en2 = ZiyuClassifier.encoder3.fit_transform(X_class_en1)
        X_prepro = np.hstack((X_num_prepro, X_class_en2))
        # 标签编码
        y_en = ZiyuClassifier.encoder4.fit_transform(y)
        return X_prepro,y_en

    def data_transform(self,X):
        """数据归一编码转换
        对输入数据归一化，编码转换
        :param X:输入的数据X
        :return:返回归一编码后的数据
        """
        X_num = X.loc[:, ZiyuClassifier.keys_num]
        X_num_prepro = ZiyuClassifier.encoder1.transform(X_num)
        X_class = X.loc[:, ZiyuClassifier.keys_class]
        X_class_en1 = ZiyuClassifier.encoder2.transform(X_class)
        X_class_en2 = ZiyuClassifier.encoder3.transform(X_class_en1)
        X_prepro=np.hstack((X_num_prepro, X_class_en2))
        return X_prepro

    def fit(self,X,y):
        """模型拟合
        对选定的模型拟合
        :param X: 输入X
        :param y: 标签y
        :return: 拟合后的模型
        """
        #model = RandomForestClassifier(n_estimators=120, min_samples_leaf=1, max_depth=12, max_features=0.4)
        self.model.fit(X, y)
        return self.model

    def predict(self,X):
        """模型预测
        利用拟合的模型预测
        :param X: 输入X
        :return: 预测的结果
        """
        return self.model.predict(X)

    def para_opt(self, X, y, param_grid, cv=5, scoring='roc_auc'):
        """网格搜索模型参数优化器
        利用cv的得分进行网格搜索优化参数
        :param X:输入X
        :param y:标签y
        :param param_grid:dict，model参数范围
        :param cv:cv
        :param scoring:评估指标
        :return:参数优化后的模型
        """
        opt_model=GridSearchCV(estimator=self.model,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scoring)
        opt_model.fit(X, y)
        print(opt_model.best_params_)
        return opt_model.best_estimator_

    def plot_learning_curve(self, name, X, y, cv=5):
        """画学习曲线
        根据cv结果画学习曲线
        :param name:标题
        :param X:输入X
        :param y: 标签y
        :param cv:cv
        :return:plt
        """
        plt=plot_learning_curve(self.model, name, X, y, ylim=None,cv=cv)
        return plt

if __name__ == "__main__":
    data_all = pd.read_csv('E:/智能运维/工单查询问题/78910月原始问题库数据_不考虑无单_all_utf8.csv', sep=',', encoding='utf8')
    test = data_all[data_all['问题触发时间'] == '9月']
    # 抽样
    train_all = data_all[data_all['问题触发时间'] != '9月']
    train = train_all  # pd.concat([train_all[train_all.自愈状态=='派单'].sample(frac=0.5,axis=0,random_state=0),train_all[train_all.自愈状态=='自愈']],axis=0,join='outer')
    print("训练样本比例为%f" % (train[train['自愈状态'] == '派单'].shape[0] / train[train['自愈状态'] == '自愈'].shape[0]))
    print("测试样本比例为%f" % (test[test['自愈状态'] == '派单'].shape[0] / test[test['自愈状态'] == '自愈'].shape[0]))
    """
    新数据来时，缺省值、异常值判断，新数据数据格式建议为dict或DataFrame，包含字段名
    """
    model=ZiyuClassifier(RandomForestClassifier(n_estimators=120,min_samples_leaf=1,max_depth=12,max_features=0.4,random_state=0))
    trainX_prepro,train_y_prepro=model.data_fit_transform(train.iloc[:,:-1],train.loc[:,'自愈状态'])
    model.fit(trainX_prepro,train_y_prepro)
    # 持久化
    joblib.dump(model,'./gongdan_ziyu.model')
    # 加载模型，预处理，预测
    mdl=joblib.load('./gongdan_ziyu.model')
    testX_prepro = mdl.data_transform(test.iloc[:, :-1])
    predict_test=mdl.predict(testX_prepro)
    # mdl.plot_learning_curve(name='RF learning_curve',X=trainX_prepro,y=train_y_prepro,cv=5)
    # print(mdl.model)
    print(metrics.confusion_matrix(ZiyuClassifier.encoder4.transform(test.iloc[:,-1]), predict_test))
    print(metrics.classification_report(ZiyuClassifier.encoder4.transform(test.iloc[:,-1]), predict_test))
    pass
