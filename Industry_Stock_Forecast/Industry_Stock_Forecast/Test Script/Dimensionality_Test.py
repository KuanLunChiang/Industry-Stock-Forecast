import pandas as pd
import numpy as np


############### Initialization ###################################################
_data = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A3\Industry-Stock-Forecast\Industry_Stock_Forecast\Industry_Stock_Forecast\Data\Data_2018.csv")
_data = _data.drop('Unnamed: 0', axis = 1)
_data.describe()

########### Class Attribute ###########################################
_colName = _data.columns.tolist()
_dataDict = {}
_responseVar = 'target'
_windowList = [80]
_paraList = [1,2,3]
_trainDict = {}
_testDict = {}

########### Variable Construction ##########################
def varCons (data, colName, target):
    df = pd.DataFrame()
    df['target'] = data[target]
    for i in colName:
        if i == target:
            df['lagTerm'] = data[target].shift(1)
        else:
            df[i] = data[i]
    df = df.dropna()
    return df

for i in _colName:
    _dataDict[i] = varCons(_data,_colName,i)

assert len(_dataDict) == len(_colName)

################ Train and Test Split ############################################
for i in _colName:
    _trainDict[i] = _dataDict[i].iloc[0:800]
    _testDict[i] = _dataDict[i].drop(_trainDict[i].index)


################### LASSO selection ########################################################

#from sklearn.svm import SVR
#mdl = SVR(kernel = 'rbf', cache_size = 2000)
#res = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _colName, regress = True, fixed = True, greedy = True, n_jobs = 4, verbose = 50, backend = 'multiprocessing')
#res.report_tuned


from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 2000)
train = _trainDict['Food']
from Time_Series.CrossValidation import rolling_Horizon
#rolling_Horizon(mdl,train,_responseVar,4,0,True,True,np.arange(1,5,1),'Lasso')
from Dimensionality_Reduction.SubsetSelection import PCA_Selection

pc = PCA_Selection(mdl,train,_responseVar)

pc.selectFeatures
pc.pcNum
pc.PC
pc.selectPC
pc.coef

mdl.fit(train)
from Time_Series.CrossValidation import rolling_Horizon
rolling_Horizon(mdl,train,_responseVar,80,0,True,True,np.arange(1,5,1),'PCA')


train.loc[1:5].drop('target',axis =1).tail(1)


from sklearn.linear_model import Lasso
from sklearn import linear_model
mdl = Lasso()
mdl == Lasso()
isinstance(mdl, linear_model)

##################### Repos ###################################
bch = tcv.benchMark()
bch.Linear_Regression(_trainDict,lasso_tune.report_tuned,_responseVar)

_reportDF = {}
for i in _colName:
    _reportDF[i] = rpt.cum_sse_report(lasso_tune.errorList[i],bch.error2[i]).reportDF
_reportDF