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
_windowList = [60,120,180,360]
_paraList = np.arange(0.00001,0.0001,0.00001)
_trainDict = {}
_testDict = {}

######### Variable Construction ##########################
def varCons (data, colName, target):
    df = pd.DataFrame()
    df['target'] = data[target]
    for i in colName:
        if i == target:
            df['lagTerm'] = data[target].shift(5)
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
rpt.outPutReport(lasso_tune,'lasso_lag5')
lasso_tune.report_tuned


################### Random Forest #####################################################################
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators = 5, max_features = len(_colName))
_paraList = np.arange(20,len(_colName),2)
rf_tune = tcv.paralell_processing(mdl,_trainDict,_responseVar,_windowList,_paraList,'C',_targetCol,True,True,True,6,50, 'multiprocessing', "None")
rpt.outPutReport(rf_tune,'randomForest_lag5')

######################### SVM #########################################
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 20000)
svm_tune = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 4, verbose = 50, backend = 'multiprocessing', dr = 'None', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(svm_tune,'SVM_lag10')
svm_tune_lasso = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = [0.0001])
rpt.outPutReport(svm_tune_lasso,'SVM_Lasso')
svm_tune_pca = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = -1, verbose = 50, backend = 'multiprocessing', dr = 'PCA', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(svm_tune_pca,'SVM_pca_test')

svm_tune_rf_lag10 = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'rf', drparam = _rfpara)
rpt.outPutReport(svm_tune_rf_lag10,'SVM_rf_lag10')



######################### KNN ############################################
from sklearn.neighbors import KNeighborsRegressor
mdl = KNeighborsRegressor()
_paraList = np.arange(1,10,1)
knn_tune =  tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'n_neighbors', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'None', drparam = np.arange(0.00001,0.0001,0.00001))
rpt.outPutReport(knn_tune,'KNN_lag5')
knn_tune_lasso =  tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'n_neighbors', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = [0.0001])
rpt.outPutReport(knn_tune_lasso,'KNN_lasso')


############################ Subset Selection ######################################################

rfinfo = pd.read_csv(r'.\Output\Window and Parameter\randomForest_lag5_winPara.csv')
lassoinfo = pd.read_csv(r'.\Output\Window and Parameter\lasso_lag5_winPara.csv')
_lassopara = {}
_rfpara = {}
for i in _targetCol:
    _rfpara[i] = int(rfinfo.loc[rfinfo.Name == i]['para'])
    _lassopara[i] = float(lassoinfo.loc[lassoinfo.Name == i]['para'])


knn_tune_lasso =  tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'n_neighbors', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = _lassopara)
knn_tune_rf =  tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'n_neighbors', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'rf', drparam = _rfpara)
rpt.outPutReport(knn_tune_lasso,'KNN_lasso_lag5')
rpt.outPutReport(knn_tune_rf,'KNN_rf_lag5')


svm_tune_lasso = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'Lasso', drparam = _lassopara)
svm_tune_rf = tcv.paralell_processing(mdl = mdl, data = _trainDict,responseVar = _responseVar, windowList = _windowList, paramList = _paraList, paraName = 'C', colName = _targetCol, regress = True, fixed = True, greedy = True, n_jobs = 6, verbose = 50, backend = 'multiprocessing', dr = 'rf', drparam = _rfpara)
rpt.outPutReport(svm_tune_lasso,'SVM_Lasso_lag5')
rpt.outPutReport(svm_tune_rf,'SVM_rf_lag5')


########################### Test Set ############################################
_knn_rf = pd.read_csv(r'./Output/Window and Parameter/KNN_rf_winPara.csv')
_knn_lasso = pd.read_csv(r'./Output/Window and Parameter/KNN_lasso_winPara.csv')
from sklearn.neighbors import KNeighborsRegressor
   
def report_cons_support(_mdlDict, _reportDict, bm, dpara, dr, i, mdl, paraName, report_tune):
    wsize = int(_reportDict.loc[report_tune['Name']==i]['Window_size'])
    para = int(_reportDict.loc[report_tune['Name']==i]['para'])
    setattr(mdl,paraName,para)
    _mdlDict[i] = tcv.rolling_Horizon(mdl,_testDict[i],_responseVar,wsize,0,True,True,dpara,dr)
    _reportDict[i]= rpt.cum_sse_report(_mdlDict[i].error2,bm.error2[i]).reportDF

def report_cons (mdl,paraName,data,report_tune,targetCol,responseVar, dpara = 0,dr = 'None'):
    _mdlDict = {}
    _reportDict = {}
    bm = tcv.benchMark()
    bm.Linear_Regression(data,report_tune,responseVar)
    for i in targetCol:
        report_cons_support(_mdlDict, report_tune, bm, dpara, dr, i, mdl, paraName, report_tune)
    return _mdlDict, _reportDict

knn = KNeighborsRegressor()
_knnDict1, _knnReport1 = report_cons(knn,'n_neighbors',_testDict,_knn_rf,_targetCol,_responseVar,[34,36,40,26,30],'rf')
rpt.plot_differential_report(_targetCol,_knnReport1,'SSEDif',3,2,'SSE Differential')
rpt.plot_differential_report(_targetCol,_knnReport1,'oosrsquare',3,2,'Out of sample R squared')


from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf', cache_size = 20000)
