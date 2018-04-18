from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np

_data = pd.read_csv(r".\Data\Data_2018.csv")
_data = _data.drop('Unnamed: 0', axis = 1)
_data.describe()

########### Class Attribute ###########################################
_colName = _data.columns.tolist()
_targetCol = ['Mach','Whlsl','BusSv','Gold','Smoke','Soda']
_dataDict = {}
_responseVar = 'target'
_windowList = [40,120,180,360]
_paraList = np.arange(0.00001,0.0001,0.00001)
_trainDict = {}
_testDict = {}

########### Variable Construction ##########################
def varCons (data, colName, target):
    df = pd.DataFrame()
    df['target'] = data[target]
    for i in colName:
        if i == target:
            df['lagTerm'] = data[target].shift(10)
        else:
            df[i] = data[i]
    df = df.dropna()
    return df

for i in _colName:
    _dataDict[i] = varCons(_data,_colName,i)

assert len(_dataDict) == len(_colName)

################ Train and Test Split ############################################
for i in _colName:
    _trainDict[i] = _dataDict[i].iloc[0:10000]
    _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

#################### LASSO ############################################################
from sklearn.linear_model import Lasso
mdl = Lasso(precompute = True, normalize = True)
lasso_tune = tcv.paralell_processing(mdl,_trainDict,_responseVar,_windowList,_paraList,'alpha',_targetCol,True,True,True,6,50, 'multiprocessing', "None")
rpt.outPutReport(lasso_tune,'lasso_lag30')
lasso_tune.report_tuned


################### Random Forest #####################################################################
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators = 5, max_features = len(_colName))
_paraList = np.arange(20,len(_colName),2)
rf_tune = tcv.paralell_processing(mdl,_trainDict,_responseVar,_windowList,_paraList,'C',_targetCol,True,True,True,6,50, 'multiprocessing', "None")
rpt.outPutReport(rf_tune,'randomForest_lag10')

######################### SVM #########################################
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 20000)
svm_tune = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 4, verbose = 50, backend = 'multiprocessing', dr = 'None', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(svm_tune,'SVM')
svm_tune_lasso = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = np.arange(0.00001,0.00002,0.00001))
rpt.outPutReport(svm_tune_lasso,'SVM_Lasso_test')
svm_tune_pca = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = -1, verbose = 50, backend = 'multiprocessing', dr = 'PCA', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(svm_tune_pca,'SVM_pca_test')
svm_tune_rf = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'rf', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(svm_tune_rf,'SVM_rf_test')



######################### KNN ############################################
from sklearn.neighbors import KNeighborsRegressor
mdl = KNeighborsRegressor()
_paraList = np.arange(1,10,1)
knn_tune =  tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'n_neighbors', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'None', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(knn_tune,'KNN')

