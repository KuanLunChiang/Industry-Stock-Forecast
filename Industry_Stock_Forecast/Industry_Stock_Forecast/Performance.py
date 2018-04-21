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
_paraList = np.arange(0.0001,0.001,0.0001)
_trainDict = {}
_testDict = {}

######### Variable Construction ##########################
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



########################### Model Preparation ############################################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from timeit import default_timer as timer

def report_cons_support(_mdlDict, _reportDict, bm, dpara, dr, i, mdl, paraName, report_tune):
    wsize = int(_reportDict.loc[report_tune['Name']==i]['Window_size'])
    if isinstance(mdl,RandomForestRegressor) or isinstance(mdl,KNeighborsRegressor):
        para = int(_reportDict.loc[report_tune['Name']==i]['para'])
    else:
        para = float(_reportDict.loc[report_tune['Name']==i]['para'])
    setattr(mdl,paraName,para)
    mdlDict = tcv.rolling_Horizon(mdl,_testDict[i],_responseVar,wsize,0,True,True,dpara,dr)
    reportDict= rpt.cum_sse_report(mdlDict.error2,bm.error2[i]).reportDF
    return mdlDict, reportDict

def report_cons (mdl,paraName,data,report_tune,targetCol,responseVar, dpara = 0,dr = 'None'):
    mdlDict = {}
    reportDict = {}
    bm = tcv.benchMark()
    bm.Linear_Regression(data,report_tune,responseVar)
    for i in targetCol:
        if dr != 'None':
            dpr = [dpara[i]]
        else: 
            dpr = 0
        tempMdl, tempReport = report_cons_support(_mdlDict, report_tune, bm, dpr, dr, i, mdl, paraName, report_tune)
        mdlDict[i] = tempMdl
        reportDict[i] = tempReport
    return mdlDict, reportDict



########################## No Feature Selection ####################################################################
_lasso = pd.read_csv(r'./Output/Window and Parameter/lasso_lag10_winPara.csv')
_rf = pd.read_csv(r'./Output/Window and Parameter/RandomForest_lag10_winPara.csv')
_knn = pd.read_csv(r'./Output/Window and Parameter/KNN_lag10_winPara.csv')
_svm = pd.read_csv(r'./Output/Window and Parameter/SVM_lag10_winPara.csv')


knn = KNeighborsRegressor()
svm = SVR(cache_size= 10000)
rf = RandomForestRegressor(n_estimators=5)
lasso = Lasso(precompute=True)
_tuneOrder = ['lasso','rf','knn','svm']
_cvList = [_lasso,_rf, _knn, _svm]
_mdlList = [lasso, rf, knn, svm]
_paraNameList = ['alpha','max_features','n_neighbors','C']
_rptDict = {}
_mdlDict = {}


for l in [1,5,10]:
    for i in _colName:
        _dataDict[i] = varCons(_data,_colName,i,l)
    assert len(_dataDict) == len(_colName)

    for i in _colName:
        _trainDict[i] = _dataDict[i].iloc[0:10000]
        _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

    for i in range(len(_mdlList)):
        start = timer()
        mdlDict, reportDict = report_cons(mdl = _mdlList[i], paraName=_paraNameList[i],data=_testDict,report_tune=_cvList[i],targetCol=_targetCol,responseVar=_responseVar)
        _rptDict[_tuneOrder[i] + '_lag'+l] = reportDict
        _mdlDict[_tuneOrder[i]+ '_lag'+l] = mdlDict
        end = timer()
        print(end - start)
        print(_tuneOrder[i])

for i in _tuneOrder:
    rpt.plot_differential_report(_targetCol,_rptDict[i],'SSEDif',2,3,'SSE Diffferential '+i)






#################################### Feature Selection ###########################################################
_knn_rf = pd.read_csv(r'./Output/Window and Parameter/KNN_rf_winPara.csv')
_knn_lasso = pd.read_csv(r'./Output/Window and Parameter/KNN_lasso_winPara.csv')
_svm_lasso = pd.read_csv(r'./Output/Window and Parameter/svm_lasso_winPara.csv')
_svm_rf = pd.read_csv(r'./Output/Window and Parameter/svm_rf_winPara.csv')
rfinfo = pd.read_csv(r'./Output/Window and Parameter/RandomForest_Tune_winPara.csv')
lassoinfo = pd.read_csv(r'./Output/Window and Parameter/lasso_winPara.csv')

_fsrptDict = {}
_fsmdlDict = {}
_fsTune = ['knn_lasso']
_fsOrder = ['Lasso']
_lassopara = {}
_rfpara = {}
_mdlList = [knn]
_paraNameList = ['n_neighbors'] 
_cvList = [_knn_lasso]


for i in _targetCol:
    _rfpara[i] = int(rfinfo.loc[rfinfo.Name == i]['para'])
    _lassopara[i] = float(lassoinfo.loc[lassoinfo.Name == i]['para'])

for i in range(len(_fsTune)):
    start = timer()
    if _fsOrder == 'Lasso':
        dparam = _lassopara
    else:
        dparam = _rfpara
    mdlDict, reportDict = report_cons(mdl = _mdlList[i], paraName=_paraNameList[i],data=_testDict,report_tune=_cvList[i],targetCol=_targetCol,responseVar=_responseVar,dpara = dparam ,dr =_fsOrder[i] )
    _fsrptDict[_fsOrder[i]] = reportDict
    _fsmdlDict[_fsOrder[i]] = mdlDict
    end = timer()
    print(end - start)
    print(_tuneOrder[i])

for i in _fsOrder:
    rpt.plot_differential_report(_targetCol,_fsrptDict[i],'SSEDif',2,3,'SSE Diffferential '+i)