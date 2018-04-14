from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np

_data = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A3\Industry-Stock-Forecast\Industry_Stock_Forecast\Industry_Stock_Forecast\Data\Data_2018.csv")
_data = _data.drop('Unnamed: 0', axis = 1)
_data.describe()

########### Class Attribute ###########################################
_colName = _data.columns.tolist()
_dataDict = {}
_responseVar = 'target'
_windowList = [80]
_paraList = np.arange(0.0001,0.0002,0.0001)
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
    _trainDict[i] = _dataDict[i].iloc[0:100]
    _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

#################### LASSO ############################################################
from sklearn.linear_model import Lasso
mdl = Lasso(precompute = True, normalize = True)
lasso_tune = tcv.paralell_processing(mdl,_trainDict,_responseVar,_windowList,_paraList,'alpha',_colName,True,True,True,4,50, 'multiprocessing', "None")
c = lasso_tune.report_tuned
c.loc[c.Name == 'Food'].Window_size

mx = lasso_tune.report_tuned.para.max()
mn = lasso_tune.report_tuned.para.min()
_lasso_range = np.arange(mn,mx,mn)
if len(_lasso_range)<1:
    _lasso_range = [mn]

################### Random Forest #####################################################################
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators = 10, max_features = len(_colName))
_paraList = np.arange(1,len(_colName),1)
rf_tune = tcv.paralell_processing(mdl,_trainDict,_responseVar,_windowList,_paraList,'C',_colName,True,True,True,4,50, 'multiprocessing', "None")





######################### SVM #########################################
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 10000)
res = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _colName, regress = True, fixed = True, greedy = True, n_jobs = 4, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = np.arange(0.00001,0.0001,0.00001))
res.report_tuned
