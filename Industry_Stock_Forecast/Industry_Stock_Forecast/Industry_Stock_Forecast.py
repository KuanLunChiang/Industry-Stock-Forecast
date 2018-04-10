from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np

_data = pd.read_csv(r"C:\Users\user\Desktop\Assignment\Industry-Stock-Forecast\Industry_Stock_Forecast\Industry_Stock_Forecast\Data\Data_2018.csv")
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
<<<<<<< HEAD
    _trainDict[i] = _dataDict[i].iloc[0:1000]
=======
    _trainDict[i] = _dataDict[i].iloc[0:100]
>>>>>>> d9f526e3d02fc6f241218244f9b2d95998afecdd
    _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

######################### SVM #########################################
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 10000)
<<<<<<< HEAD
res = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _colName, regress = True, fixed = True, greedy = True, n_jobs = 4, verbose = 50, backend = 'multiprocessing', dr = 'None')
=======
res = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _colName, regress = True, fixed = True, greedy = True, n_jobs = 8, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = np.arange(0,1,1))
>>>>>>> d9f526e3d02fc6f241218244f9b2d95998afecdd
res.report_tuned
res.coefList['Food']

