from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.externals.joblib import Parallel, delayed
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
def varCons (data, colName, target, lagnum):
    df = pd.DataFrame()
    for i in colName:
        if i == target:
            df['lagTerm'] = data[target].shift(lagnum)
        else:
            df[i] = data[i]
    df['target'] = data[target]
    df = df.dropna()
    return df

########################### Model Preparation ############################################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, LinearSVR
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
    return (mdlDict, reportDict)

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
        tempMdl, tempReport = report_cons_support(mdlDict, report_tune, bm, dpr, dr, i, mdl, paraName, report_tune)
        mdlDict[i] = tempMdl
        reportDict[i] = tempReport
    mdlDict = tempMdl
    reportDict = tempReport
    return mdlDict, reportDict


def outputReport(mdl ,name):
    coefDF = mdl.coefSelection
    errorDF = mdl.error2
    json_output(coefDF,'.\\Output\\Coefficient\\'+name+'_coefficient'+'.json')
    json_output(errorDF,'.\\Output\\Error List\\'+name+'_errorList'+'.json')


def json_output (data, output):
    import json
    with open(output, 'w') as outfile:
        json.dump(data,outfile)



########################## No Feature Selection ####################################################################
knn = KNeighborsRegressor()
svm = SVR(cache_size= 10000)
rf = RandomForestRegressor(n_estimators=10)
lasso = Lasso(precompute=True)
tuneOrder = ['knn','lasso','rf','svm']
_mdlList = [knn,lasso, rf, svm]
_paraNameList = ['n_neighbors','alpha','max_features','C']
_rptDict = {}
_mdlDict = {}

for l in [1,5,10]:
    ttlStart = timer()
    _lasso = pd.read_csv(r'./Output/Window and Parameter/lasso_lag'+str(l)+'_winPara.csv')
    _rf = pd.read_csv(r'./Output/Window and Parameter/randomForest_lag'+str(l)+'_winPara.csv')
    _knn = pd.read_csv(r'./Output/Window and Parameter/knn_lag'+str(l)+'_winPara.csv')
    _svm = pd.read_csv(r'./Output/Window and Parameter/svm_lag'+str(l)+'_winPara.csv')
    _cvList = [_knn,_lasso,_rf,_svm]

    for i in _colName:
        _dataDict[i] = varCons(_data,_colName,i,l)
        _trainDict[i] = _dataDict[i].iloc[0:10000]
        _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

    for i in range(len(_mdlList)):
        start = timer()
        mdlDict, reportDict = report_cons(mdl = _mdlList[i], paraName=_paraNameList[i],data=_testDict,report_tune=_cvList[i],targetCol=_targetCol,responseVar=_responseVar)
        _rptDict[tuneOrder[i] + '_lag'+str(l)] = reportDict
        _mdlDict[tuneOrder[i]+ '_lag'+str(l)] = mdlDict
        end = timer()
        print(end - start)
        print(tuneOrder[i])
    ttlend = timer()
    print(ttlend - ttlStart)




#################################### Feature Selection ###########################################################
_fsrptDict = {}
_fsmdlDict = {}
_fsTune = ['knn_lasso','knn_rf','svm_lasso','svm_rf']
_fsOrder = ['Lasso','rf','Lasso','rf']
_lassopara = {}
_rfpara = {}
_mdlList = [knn,knn,svm,svm]
_paraNameList = ['n_neighbors','n_neighbors','C','C'] 

for l in [1,5,10]:
    ttlStart = timer()
    _knn_rf = pd.read_csv(r'./Output/Window and Parameter/knn_rf_lag'+str(l)+'_winPara.csv')
    _knn_lasso = pd.read_csv(r'./Output/Window and Parameter/knn_lasso_lag'+str(l)+'_winPara.csv')
    _svm_lasso = pd.read_csv(r'./Output/Window and Parameter/svm_lasso_lag'+str(l)+'_winPara.csv')
    _svm_rf = pd.read_csv(r'./Output/Window and Parameter/svm_rf_lag'+str(l)+'_winPara.csv')
    rfinfo = pd.read_csv(r'./Output/Window and Parameter/randomForest_lag'+str(l)+'_winPara.csv')
    lassoinfo = pd.read_csv(r'./Output/Window and Parameter/lasso_lag'+str(l)+'_winPara.csv')
    _cvList = [_knn_lasso,_knn_rf,_svm_lasso,_svm_rf]
    for i in _colName:
        _dataDict[i] = varCons(_data,_colName,i,l)
    assert len(_dataDict) == len(_colName)

    for i in _colName:
        _trainDict[i] = _dataDict[i].iloc[0:10000]
        _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

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
        _fsrptDict[_fsTune[i]+'_lag'+str(l)] = reportDict
        _fsmdlDict[_fsTune[i]+'_lag'+str(l)] = mdlDict
        end = timer()
        print(end - start)
        print(_fsTune[i])
    for i in _fsrptDict:
        rpt.plot_differential_report(_targetCol,_fsrptDict[i],'SSEDif',2,3,'SSE Diffferential '+i)
    ttlend = timer()
    print(ttlend - ttlStart)



########################### Plots and Stats #############################################################
_fsmdlDict
_fsrptDict
cpymdl = _mdlDict.copy()
cpyrpt = _rptDict.copy()
cpfsmdl = _fsmdlDict.copy()
cpfsrpt = _fsrptDict.copy()

_fsrptDict['svm_rf_lag1']['Soda']
_rptDict['svm_lag1']['Soda']

rmseMx = pd.DataFrame(columns=['Mdl','Ind','Window Size','RMSE'])
bchrmseMx = pd.DataFrame(columns=['Mdl','Ind','Window Size','RMSE'])
for i in _fsmdlDict:
    for j in _targetCol:
        rmseMx = rmseMx.append({'Mdl':i,'Ind':j,'Window Size':_fsmdlDict[i][j].wsize,'RMSE':np.sqrt(np.mean(_fsrptDict[i][j].MdlError))},ignore_index= True)
        bchrmseMx = bchrmseMx.append({'Mdl':i,'Ind':j,'Window Size':_fsmdlDict[i][j].wsize,'RMSE':np.sqrt(np.mean(_fsrptDict[i][j].BenchError))},ignore_index= True)
for i in _mdlDict:
    for j in _targetCol:
        rmseMx = rmseMx.append({'Mdl':i,'Ind':j,'Window Size':_mdlDict[i][j].wsize,'RMSE':np.sqrt(np.mean(_rptDict[i][j].MdlError))},ignore_index= True)
        bchrmseMx = bchrmseMx.append({'Mdl':i,'Ind':j,'Window Size':_mdlDict[i][j].wsize,'RMSE':np.sqrt(np.mean(_rptDict[i][j].BenchError))},ignore_index= True)
        

rmseMx.pivot(index = 'Ind',columns = 'Mdl',values = 'RMSE').to_csv('.\Output\RMSE Matrix_pt.csv')
rmseMx.to_csv('.\Output\RMSE Matrix.csv')
bchrmseMx.pivot(index = 'Ind',columns = 'Mdl',values = 'RMSE').to_csv('.\Output\Bench RMSE Matrix.csv')

for i in _fsrptDict:
    rpt.plot_differential_report(_targetCol,_fsrptDict[i],'SSEDif',2,3,'SSE Diffferential '+i)

for i in _rptDict:
    rpt.plot_differential_report(_targetCol,_rptDict[i],'SSEDif',2,3,'SSE Diffferential '+i)


_plotDict = {}
_lagDict = {}
findList = ['lasso','rf','svm','knn']
findListFS = ['svm_lasso','svm_rf','knn_lasso','knn_rf']
lagList = ['_lag1','_lag5','_lag10']
for k in lagList:
    for j in _targetCol:
        plotFrame = pd.DataFrame()
        for i in findList:
            find = i + k
            plotFrame[i] = _rptDict[find][j].oosrsquare
        for ind in findListFS:
            find = ind + k
            plotFrame[ind] = _fsrptDict[find][j].oosrsquare
        _plotDict[j] = plotFrame.copy()
    _lagDict[k] = _plotDict.copy()



for j in _targetCol:
    #plt.axhline(y=0, color='black', linestyle='-')
    _lagDict['_lag10'][j].plot(figsize = (10,10))
    plt.title(j)
    
############################ Feature Map ######################################################
#for i in cpfsmdl:
#    if i not in ['knn_lasso_lag1','knn_lasso_lag5','knn_lasso_lag10']:
#        _fsmdlDict[i] = cpfsmdl[i].copy()


import seaborn as sns
fdmdlList = ['knn_lasso_lag1','knn_lasso_lag5','knn_lasso_lag10','knn_rf_lag1', 'knn_rf_lag5', 'knn_rf_lag10']
fdmdlList = ['lasso_lag1']


for fdmd in fdmdlList:
    print(fdmd)
    for itd in _targetCol:
        coefPD = pd.DataFrame(columns = _colName+['lagTerm'])
        fdmdl = _mdlDict[fdmd][itd].coefSelection
        for i in range(len(fdmdl)):
            tempDict = {}
            for j in fdmdl[i]:
                tempDict[j] = 1
            coefPD = coefPD.append(tempDict,ignore_index= True)
        sns.set_style('ticks')
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        hm = sns.heatmap(coefPD.transpose(), cmap="YlGnBu").set_title(itd)
        plt.show()

for j in _targetCol:
    for i in _fsmdlDict['knn_lasso_lag1'][j].coefSelection:
        if len(_fsmdlDict['knn_lasso_lag1'][j].coefSelection[i]) <=48:
            print(len(_fsmdlDict['knn_lasso_lag1'][j].coefSelection[i]))



###########################################################################################
bm = tcv.benchMark()
bmDict = {}
for l in [1]:
    for j in _colName:
        _dataDict[j] = varCons(_data,_colName,j,l)
        _trainDict[j] = _dataDict[j].iloc[0:10000]
        _testDict[j] = _dataDict[j].drop(_trainDict[j].index)
    _lasso = pd.read_csv(r'./Output/Window and Parameter/lasso_lag'+str(l)+'_winPara.csv')
    _rf = pd.read_csv(r'./Output/Window and Parameter/randomForest_lag'+str(l)+'_winPara.csv')
    _knn = pd.read_csv(r'./Output/Window and Parameter/knn_lag'+str(l)+'_winPara.csv')
    _svm = pd.read_csv(r'./Output/Window and Parameter/svm_lag'+str(l)+'_winPara.csv')
    _cvList = [_knn,_lasso,_rf,_svm]
    cvName = ['knn_lag','lasso_lag','rf_lag','svm_lag']
    for rt in range(len(_cvList)):
        bmDict[cvName[rt]+l] = bm.Linear_Regression(_testDict,_cvList[rt],_responseVar)


for i in _mdlDict:
    mdlDict = _mdlDict[i]
    reportDict = {}
    for j in _targetCol:
        reportDict[j]= rpt.cum_sse_report(mdlDict[j].error2,bmDict[i][j]).reportDF
    _rptDict[i] = reportDict.copy()


for i in bmDict:
    print(i)