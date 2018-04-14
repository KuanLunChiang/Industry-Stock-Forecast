
import pandas as pd
from Dimensionality_Reduction.SubsetSelection import *
import numpy as np
from sklearn.externals.joblib import Parallel, delayed



class Feature_Selection_Tune (object):
    def __init__(self,mdl, data, responseVar = 'target' ,para = np.arange(0.0001,0.001,0.0001), dr = 'Lasso'):
        if dr == 'Lasso':
            sf = Lasso_Selection(data, para, response = responseVar)
            self.selectFeatures = [responseVar] + sf.selectFeatures.tolist()
            if len(self.selectFeatures) > 1:
                self.data = data[self.selectFeatures]
            else:
                self.data = data
        elif dr == 'rf':
            sf = RandomForest_Selection(data,para, response = responseVar)
            self.selectFeatures = [responseVar] + sf.selectFeatures.tolist()
            self.data = data[self.selectFeatures]
        elif dr == 'PCA':
            sf = PCA_Selection(mdl,data, responseVar)
            self.selectFeatures = [responseVar] + sf.selectFeatures
            self.data = sf.coef

        elif dr == 'None':
            self.data = data



class rolling_Horizon(object):
    """ 
    The core class that performs the time sereis cross validation. 
    There are two different versions provided in this function: regression and classification.
    The rolling horizon also has two versions: fixed size and expanding window.
    Classification is based on the prediction results of the regression function, and then classified by its qualtitiles.
    """



    def __init__(self, mdl, data, responseVar ,wsize=4 , startInd=0, regress = True, fixed = True, para = np.arange(0.0001,0.001,0.0001), dr = 'None'):
        from Time_Series.CrossValidation import Feature_Selection_Tune
        self.error2 = []
        self.prdList = []
        self.wsize = wsize
        self.startInd = startInd
        self.coefSelection = {}
        for i in range(startInd,len(data)):
            rlg = mdl
            testx, testy, trainx, trainy = self.data_prep(data, dr, fixed, i, mdl, para, responseVar, startInd, wsize)
            rlg.fit(trainx,trainy)
            prd = rlg.predict(testx)
            self.coefSelection[i]= trainx.columns.tolist()
            if regress:
                self.error2.append(((testy[0] - prd[0]) ** 2))
                self.prdList.append(prd[0])
            else:
                testy[0] = self.percentile_transform_three_bin(data[responseVar],testy[0])
                self.prdList.append(self.percentile_transform_three_bin(trainy,prd[0]))
                prd = self.percentile_transform_three_bin(trainy,prd[0])
                if testy[0] - prd == 0:
                    self.error2.append(0)
                else:
                    self.error2.append(1)
            if i + wsize + 1 == len(data):
                break
        assert len(data) == self.wsize + len(self.error2) + self.startInd
        assert len(self.error2) == len(data) - (self.startInd + self.wsize)


    def data_prep(self, data, dr, fixed, i, mdl, para, responseVar, startInd, wsize):
        if fixed:
            if dr == 'PCA':
                sfdata = Feature_Selection_Tune(mdl,data[i:i + wsize+1],responseVar,para,dr).data
                trainx = sfdata.head(wsize).drop(responseVar, axis = 1)
                trainy = sfdata.head(wsize)[responseVar]
                testx = data.iloc[0:wsize+1].drop(responseVar, axis = 1).tail(1).copy()
                testy = data.iloc[0:wsize+1][responseVar].values.copy()
            else:
                sfdata = Feature_Selection_Tune(mdl,data[i:i + wsize+1],responseVar,para,dr).data
                trainx = sfdata.head(wsize).drop(responseVar, axis = 1)
                trainy = sfdata.head(wsize)[responseVar]
                testx = sfdata.tail(1).drop(responseVar, axis = 1).copy()
                testy = sfdata.tail(1)[responseVar].values.copy()
            
        else:
            if dr == "PCA":
                sfdata = Feature_Selection_Tune(mdl, data[startInd:i + wsize+1],responseVar,para,dr).data
                trainx = sfdata.drop(responseVar, axis = 1)
                trainy = sfdata[responseVar]
                testx = data.iloc[0:wsize+1].drop(responseVar, axis = 1).tail(1).copy()
                testy = data.iloc[0:wsize+1][responseVar].values.copy()
            else:
                sfdata = Feature_Selection_Tune(mdl, data[startInd:i + wsize+1],responseVar,para,dr).data
                trainx = sfdata.head(wsize).drop(responseVar, axis = 1)
                trainy = sfdata.head(wsize)[responseVar]
                testx = sfdata.tail(1).drop(responseVar, axis = 1).copy()
                testy = sfdata.tail(1)[responseVar].values.copy()
        return testx, testy, trainx, trainy


    def percentile_transform_three_bin (self,data, testd, lb = 0.33, ub = 0.66):
        res = 0
        tempdata = data.copy()
        lower = tempdata.quantile(lb)
        upper = tempdata.quantile (ub)
        if testd < lower:
            res = 1
        elif testd > upper:
            res = 3
        else:
            res = 2
        return res

 


class rolling_cv(object):
    """
    This class provide a grid search selection for window size based on the time series cross validation.
    """
    from Time_Series.CrossValidation import rolling_Horizon


    def __init__(self, data, responsVar ,mdl,windowList, regress = True, fixed = True, para = np.arange(0.0001,0.001,0.0001), dr = 'None', drparam = np.arange(0.0001,0.001,0.0001)):
        rmse = {}
        wsize = 0
        for w in windowList:
             rh = rolling_Horizon(mdl = mdl, data = data, responseVar = responsVar ,wsize = w, startInd = 0,regress = regress, fixed = fixed, para = drparam, dr = dr)
             error2 = rh.error2
             prd = rh.prdList
             rmse[w] = np.sqrt(np.mean(np.cumsum(error2)))
        wsize = min(rmse, key = rmse.get)
        rh = rolling_Horizon(mdl = mdl, data = data,responseVar = responsVar, wsize= wsize, startInd = 0 , regress = regress, fixed = fixed , para = drparam, dr = dr)
        se = rh.error2
        prdList = rh.prdList
        rmse = np.sqrt(np.mean(np.cumsum(se)))
        self.rmse = rmse
        self.bestWindow = wsize
        self.performance = rmse
        self.prdList = prdList
        self.error2 = se
        self.windowList = windowList
        self.isRegress = regress
        self.isFixed = fixed
        self.coefSelection = rh.coefSelection


class grid_tune_parameter (object):

    """
    This class implments a grid search for the window size and one parameter based on the time series cross validation.
    """

    from Time_Series.CrossValidation import rolling_cv
    
    def __init__(self, mdl, data, responseVar, window, paramList , paramName, regress = True, fixed = True, dr = 'None', drparam = np.arange(0.0001,0.001,0.0001)):
        tuneSelection = pd.DataFrame(columns = ['param','window','rmse'])
        sse = {}
        prdList = {}
        coefSelection = {}
        for i in paramList:
            sizeselect = {}
            setattr(mdl,paramName,i)
            rc = rolling_cv(data, responseVar, mdl,window, regress,fixed = fixed, para = paramList, dr = dr, drparam = drparam)
            se = rc.error2
            rmse = rc.rmse 
            wsize = rc.bestWindow 
            prdList[i] = rc.prdList
            tuneSelection = tuneSelection.append({'param':i,'window':wsize,'rmse':rmse},ignore_index=True)
            sse[i] = se
            coefSelection[i] = rc.coefSelection
        self.tuned = tuneSelection.iloc[tuneSelection.rmse.idxmin()]
        self.para = self.tuned.param
        self.wsize = self.tuned.window
        self.error2 = sse[self.tuned.param]
        self.prdList = prdList[self.tuned.param]
        self.coefSelection = coefSelection[self.tuned.param]
        

class sequential_grid_tune (object):
    """
    This class uses greedy approach to obtain the local optimal point. 
    Given a starting parameter number, the function will use the number to get the best window size.
    Then, the best window size will be used to get the local optimal parameter number.
    """

    from Time_Series.CrossValidation import grid_tune_parameter

    def __init__(self, data, responseVar ,mdl, window , paramList, paramName , startPara = 1 , regress = True, fixed = True, dr = 'None', drparam = np.arange(0.0001,0.001,0.0001)):

        windowSelect = grid_tune_parameter(mdl,data, responseVar,window,[startPara],paramName,regress, fixed = fixed, dr = dr, drparam = drparam)
        size = [int(windowSelect.wsize)]
        paraSelect = grid_tune_parameter(mdl,data,responseVar,size,paramList,paramName, regress, fixed = fixed, dr = dr, drparam = drparam)
        self.tuned = paraSelect.tuned
        self.para = paraSelect.para
        self.wsize = paraSelect.wsize
        self.error2 = paraSelect.error2
        self.prdList = paraSelect.prdList
        self.coefSelection = paraSelect.coefSelection


class paralell_processing (object):
    """
    This class provides a threading based parallel computing interface for both sequential_grid_tune and grid_tune_paramenter functions.
    """

    from Time_Series.CrossValidation import sequential_grid_tune, grid_tune_parameter
    from sklearn.externals.joblib import Parallel, delayed
    def __init__(self, mdl, data, responseVar ,windowList, paramList, paraName,colName ,regress = True, fixed = True, greedy = True, n_jobs = -4, verbose = 50, backend = 'multiproccessing', dr = 'None', drparam = np.arange(0.0001,0.001,0.0001)):
        errorList = {}
        wisize = {}
        prdList= {}
        coefList = {}
        colName = colName
        report = Parallel(n_jobs = n_jobs, verbose = verbose, backend = backend)(delayed(self.paralell_support)(i,mdl,data, responseVar,regress,windowList,paramList, paraName, fixed, greedy, dr, drparam) for i in colName)
        for i in colName:
            errorList[i] = report[colName.index(i)]['el']
            wisize[i] =report[colName.index(i)]['tune']
            prdList[i] = report[colName.index(i)]['prd']
            coefList[i] = report[colName.index(i)]['coef']
        self.errorList = errorList
        self.wisize = wisize
        self.prdList = prdList
        self.coefList = coefList
        report_tuned = pd.DataFrame()
        for i in range(len(colName)):
            report_tuned = report_tuned.append(self.wisize[colName[i]])
        report_tuned= report_tuned.reset_index().drop('index',axis = 1)
        self.report_tuned = report_tuned

        
    def paralell_support (self,name ,mdl, data, responseVar , regress , windowList, paramList, paraName, fixed, greedy, dr, drparam):
        tune_res = pd.DataFrame()
        el = []
        name = name
        mdl = mdl
        rsp = responseVar
        if greedy:
            sq = sequential_grid_tune(data = data[name],responseVar = rsp,mdl = mdl, window = windowList, paramList = paramList, paramName = paraName, startPara = 1, regress = regress, fixed = fixed, dr = dr, drparam = drparam)
        else:
            sq = grid_tune_parameter(mdl = mdl, data = data[name], responseVar = rsp, window = windowList, paramList = paramList, paramName = paraName, regress = regress, fixed = fixed, dr = dr, drparam = drparam)
        se = sq.error2
        tuned = sq.tuned
        para = sq.para
        wsize = sq.wsize
        prdList = sq.prdList
        tune_res = tune_res.append({'Window_size': wsize, 'Name': name, 'para': para},ignore_index= True)
        el= se
        coefSelection = sq.coefSelection
        return {'tune': tune_res, 'el':el, 'prd':prdList, 'coef': coefSelection}


class benchMark (object):
    """
    This classs provides the benchmark estimators' SSE. 
    The benchmark for regression model is the rolling mean, while the classification model is the value of previous point.
    """
    def __init__(self):
        self.error2 = []
        self.prd = []

    def historical_mean (self,data, wsize):
        prd = data.rolling(wsize).mean().dropna()
        prd.drop(prd.tail(1).index, inplace = True)
        prd.index = prd.index+1
        testy = data[wsize:,]
        error = prd.subtract(testy)
        se = error.apply(lambda x: pow(x,2)).tolist()
        try:
            assert len(prd) == len(testy)
        except:
            print('different length')
        self.error2 = se
        self.prd = prd
        return se

    def classification_benchmark (self,data):
        upper = data.quantile(0.66)
        lower = data.quantile(0.33)
        datay = data.copy()
        res = data.shift(1).dropna().apply(lambda x: 1 if x < lower else (3 if x > upper else 2))
        datay = datay.apply(lambda x: 1 if x < lower else (3 if x > upper else 2))
        datay = datay[1:]
        se = datay.subtract(res).apply(lambda x: 0 if x == 0 else 1)
        se = list(se)
        self.error2 = se
        self.prd = res
        return se
    
    def Linear_Regression (self, data, tuneReport, responseVar):
        from sklearn.linear_model import LinearRegression
        from Time_Series.CrossValidation import rolling_Horizon
        lr = LinearRegression()
        self.error2 = {}
        self.prd = {}
        for i in data:
            wsize = int(tuneReport.loc[tuneReport.Name == i].Window_size)
            rh = rolling_Horizon(lr,data[i],responseVar,wsize)
            self.error2[i] = rh.error2
            self.prd[i] = rh.prdList
  
