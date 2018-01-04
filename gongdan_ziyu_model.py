# coding: utf8
import sys
import logging

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
    keys_class = ['场景要素', '覆盖类型', '问题现象', '地市', '区县', '数据来源', '业务要素']  # '覆盖场景'
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
        # 统计均值和众数
        self.mean_mode = self.__get_means_mode(X)
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
        # 数值字段
        X_num = X.loc[:, ZiyuClassifier.keys_num]
        X_num_prepro = ZiyuClassifier.encoder1.transform(X_num)
        # 名义字段编码
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
        :return: DataFrame,预测的结果
        """
        predict_res = pd.DataFrame()
        predict_res['自愈判断'] = self.model.predict(X)
        predict_res['自愈概率'] = np.max(self.model.predict_proba(X), axis=1)
        return predict_res

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

    def __get_means_mode(self, X):
        """统计均值和众数

        :param X: 输入X
        :return: DataFrame,各字段均值或众数
        """
        mode_X = X.loc[:, ZiyuClassifier.keys_class].mode(axis=0)
        mean_X = X.loc[:, ZiyuClassifier.keys_num].mean().to_frame().transpose()
        return pd.concat([mean_X, mode_X], axis=1, join='outer')

    def set_mean_mode(self, value):
        """外部设置均值和众数
        由外部设置均值和众数
        :param value: DataFrame，设置值
        :return: None
        """
        self.mean_mode = value

class DataChecker(object):
    """数据校验

    参数
    无

    属性
    类变量：
    std_data_values指标取值范围，或枚举
    实例变量：
    item_num数据记录数
    feature_num数据字段数
    missing_keys缺失关键字段
    data_exception_keys数据异常的字段
    """
    std_data_values = {'问题触发时间':[],
                       '地市':['杭州','宁波','温州','绍兴','嘉兴','湖州','丽水','金华','衢州','台州','舟山'],
                       '区县':['上城','下城','江干','拱墅','西湖','滨江','下沙','萧山','余杭','建德','富阳','临安','桐庐','淳安',
                             '海曙','江北','北仑','镇海','鄞州','奉化','余姚','慈溪','象山','宁海',
                             '鹿城','龙湾','瓯海','洞头','瑞安','乐清','永嘉','平阳','苍南','文成','泰顺',
                             '越城','绍兴','上虞','诸暨','嵊州','新昌',
                             '吴兴','南浔','德清','长兴','安吉',
                             '南湖','秀洲','海宁','平湖','桐乡','嘉善','海盐',
                             '婺城','金东','兰溪','东阳','永康','义乌','武义','浦江','磐安',
                             '柯城','衢江','江山','常山','开化','龙游',
                             '椒江','黄岩','路桥','临海','温岭','玉环','三门','天台','仙居',
                             '莲都','龙泉','青田','缙云','遂昌','松阳','云和','庆元','景宁','开发区',
                             '定海','普陀','岱山','嵊泗'],
                       '网络类型':['4G','2G'],
                       '网元要素':['基站','路段','小区','栅格'],
                       '数据来源':['SEQ','北向性能','LTE-MR','实时性能告警'],
                       '问题归类(一级)':[],
                       '问题归类(二级)':[],
                       '问题现象':['无线切换质差', 'VOLTE接通质差', 'SRVCC切换质差', 'VOLTE丢包质差', 'VOLTE掉话质差',
                                 '语音质差', '无线接通质差', '实时性能持续质差', 'CSFB回落质差', '无线掉线质差', 'RRC重建比质差',
                                 'CSFB性能质差','低速率','掉线质差','高负荷','高干扰','零流量'],
                       '问题类型':[],
                       '类别要素':['互操作','感知','质量','负荷','结构'],
                       '是否追加':['是','否'],
                       '主指标(事件)':[],
                       '主指标表征值':[-110,700],
                       '处理优先级':['中','高'],
                       '目前状态':['待接入','归档','人工关闭','已接入'],
                       '是否为FDD站点':['是','否'],
                       '是否实时工单已派单':['是','否'],
                       '是否指纹库智能分析系统运算':['是','否'],
                       '是否列为白名单':['是','否'],
                       '是否为性能交维站点':['是','否'],
                       '是否质检通过':['是','否','未质检'],
                       '资管生命周期状态':['工程','现网','维护','设计','在网'],
                       '劣化次数':[1,31],
                       '告警触发次数':[1,300],
                       '日均流量(GB)':[0.0,0.27],
                       '业务要素':['数据','语音'],
                       '触发要素':['劣于门限','异常事件','人工创造'],
                       '场景要素':['高速', '海域', '室外', '普铁', '室分', '地铁', '山区', '小微站', '高校', '高铁', '全网'],
                       '覆盖类型':['室内','室外'],
                       '覆盖场景':['工业园区', '写字楼', '别墅群', '低层居民区', '高铁', '医院', '高速公路', '村庄',
                                 '长途汽车站', '高层居民区', '机场', '其他', '星级酒店', '航道', '党政军机关', '广场公园',
                                 '乡镇', '商业中心', '火车站', '体育场馆', '山农牧林', '近水近海域', '高校', '国道省道',
                                 '地铁', '郊区道路', '风景区', '公墓', '普铁', '中小学', '集贸市场', '码头', '城区道路',
                                 '边境小区', '企事业单位', '城中村', '会展中心', '休闲娱乐场所'],
                       '二级场景':[],
                       '时间维度':['天'],
                       '中心经度':[118.037,123.143],
                       '中心维度':[27.231,31.176],
                       '工单类型':['集中质量分析工单'],
                       'TAC(LAC)':[22148,26840]}

    def __init__(self):
        self.item_num = 0
        self.feature_num = 0
        self.missing_keys = []
        self.data_exception_keys = []

    def __null_process(self, data, nan_fill_data):
        """数据空值填充
        根据提供的fill_data填充空值数据
        :param data:待校验的数据
        :param fill_data:各字段默认的填充值
        :return:None
        """
        if data.isnull().any().any() == True:
            nan_fill_data_dict = nan_fill_data.to_dict(orient='record')[0]
            data.fillna(nan_fill_data_dict, inplace=True)
        else:
            pass

    def data_check(self, data = pd.DataFrame(), nan_fill_data = pd.DataFrame()):
        """数据校验
        根据预设的范围判断数据是否异常，先对数据空值填充，再校验
        :param data:待校验的数据
        :param nan_fill_data:各字段默认的填充值
        :return:数据状态
        """
        logger = logging.getLogger("ZiyuLogging")
        self.item_num,self.feature_num = data.shape
        if self.item_num == 0 or self.feature_num == 0:
            # print("The file has no data!")
            return 1
        else:
            for key in ZiyuClassifier.keys_num + ZiyuClassifier.keys_class:
                try:
                    data.loc[:,key]
                except:
                    self.missing_keys.append(key)
                else:
                    pass
            if len(self.missing_keys) != 0:
                # print(self.missing_keys)
                return 2
            else:
                data_ava = data.loc[:,ZiyuClassifier.keys_num + ZiyuClassifier.keys_class]
                self.__null_process(data_ava, nan_fill_data)
                # 数值是否在合理范围
                for key in ZiyuClassifier.keys_num:
                    if True in list(data_ava.loc[:,key] < DataChecker.std_data_values[key][0])\
                            or True in list(data_ava.loc[:,key] > DataChecker.std_data_values[key][1]):
                        self.data_exception_keys.append(key)
                    else:
                        pass
                # 标称字段是否集合元素
                for key in ZiyuClassifier.keys_class:
                    if False in data_ava.loc[:,key].map(lambda x: x in DataChecker.std_data_values[key]).tolist():
                        self.data_exception_keys.append(key)
                    else:
                        pass
                if len(self.data_exception_keys) != 0:
                    # print(self.data_exception_keys)
                    return 3
                else:
                    return 0

class ZiyuLogging(object):
    """日志记录
    记录调试和校验日志
    """
    @staticmethod
    def config(logger = logging.getLogger("ZiyuLogging")):
        """日志配置

        :param logger:创建Logging对象
        :return:None
        """
        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')
        # 文件日志
        file_handler = logging.FileHandler("ziyu_mode.log",encoding='utf8')
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter  # 也可以直接给formatter赋值
        # 为logger添加的日志处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.INFO)

    def test_logging1(self):
        logger = logging.getLogger("ZiyuLogging")
        logger.info("Test ZiyuLogging")

    @classmethod
    def test_logging2(cls):
        cls.config(logger= logging.getLogger("OtherLogging"))
        logger = logging.getLogger("OtherLogging")
        logger.error("Test OtherLogging")

if __name__ == "__main__":
    data_all = pd.read_csv('E:/智能运维/工单查询问题/78910月原始问题库数据_不考虑无单_all_utf8.csv', sep=',', encoding='utf8')
    test = data_all[data_all['问题触发时间'] == '9月'].reset_index(drop=True)
    # 抽样
    train_all = data_all[data_all['问题触发时间'] != '9月'].reset_index(drop=True)
    train = train_all  # pd.concat([train_all[train_all.自愈状态=='派单'].sample(frac=0.5,axis=0,random_state=0),train_all[train_all.自愈状态=='自愈']],axis=0,join='outer')
    print("训练样本比例为%f" % (train[train['自愈状态'] == '派单'].shape[0] / train[train['自愈状态'] == '自愈'].shape[0]))
    print("测试样本比例为%f" % (test[test['自愈状态'] == '派单'].shape[0] / test[test['自愈状态'] == '自愈'].shape[0]))
    # 创建模型
    model=ZiyuClassifier(RandomForestClassifier(n_estimators=120,min_samples_leaf=1,max_depth=12,max_features=0.4,random_state=0))
    trainX_prepro,train_y_prepro=model.data_fit_transform(train.iloc[:,:-1],train.loc[:,'自愈状态'])
    model.fit(trainX_prepro,train_y_prepro)
    # 日志开启
    ZiyuLogging.config(logger = logging.getLogger("ZiyuLogging"))
    logger = logging.getLogger("ZiyuLogging")
    # 持久化
    joblib.dump(model,'./gongdan_ziyu.model')
    # 加载模型，预处理，预测
    mdl=joblib.load('./gongdan_ziyu.model')
    # 读取数据
    """
    新数据来时，缺省值、异常值判断，新数据数据格式建议为dict或DataFrame，包含字段名
    """
    # 添加简单校验规则
    nan_fill_data = mdl.mean_mode
    data_checker = DataChecker()
    data_status = data_checker.data_check(test,nan_fill_data)
    if data_status != 0:
        if data_status == 1:
            logger.info("The file has no data!")
            pass
        elif data_status == 2:
            logger.info('missing_keys')
            logger.info(data_checker.missing_keys)
            pass
        else:
            logger.info('exception_keys')
            logger.info(data_checker.data_exception_keys)
            pass
    else:
        # 数据转换
        testX_prepro = mdl.data_transform(test.iloc[:, :-1])
        # 预测
        predict_test=mdl.predict(testX_prepro)
        # mdl.plot_learning_curve(name='RF learning_curve',X=trainX_prepro,y=train_y_prepro,cv=5)
        # print(mdl.model)
        # print(metrics.confusion_matrix(ZiyuClassifier.encoder4.transform(test.iloc[:,-1]), predict_test['自愈状态']))
        # print(metrics.classification_report(ZiyuClassifier.encoder4.transform(test.iloc[:,-1]), predict_test['自愈状态']))
        predict_test['自愈判断'] = ZiyuClassifier.encoder4.inverse_transform(predict_test['自愈判断'])
        # 写入csv文件
        pd.concat((test,predict_test),axis=1,join='outer').to_csv(path_or_buf='predict_res.csv',sep=',',encoding='gbk')
        pass
