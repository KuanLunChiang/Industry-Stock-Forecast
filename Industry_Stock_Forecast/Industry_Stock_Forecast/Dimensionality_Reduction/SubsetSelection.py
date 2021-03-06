import pandas as pd
import numpy as np


class SubsetSelection(object):
    """description of class"""
    def __init__(self, mdl, data,response ='target'):
        self.mdl = mdl
        self.datax = data.drop(response,axis = 1)
        self.datay = data[response]

class Lasso_Selection (SubsetSelection):
    
    def __init__(self,data, alphas = np.arange(0.001,0.01, 0.001) ,response = 'target', threshold = 0, positive = False):
        from sklearn.linear_model import Lasso
        from Time_Series.CrossValidation import grid_tune_parameter
        mdl = Lasso(normalize = True, precompute = True, fit_intercept = True, positive = positive)
        super().__init__(mdl, data, response)
        self.alpha = grid_tune_parameter(mdl,data,response,[len(data)-1],alphas,'alpha').para
        mdl = Lasso(normalize = True, precompute = True, fit_intercept = True, positive = positive, alpha = self.alpha)
        mdl.fit(self.datax, self.datay)
        self.threshold = threshold
        self.coef = self.mdl.coef_
        self.coefN = self.datax.columns.tolist()
        self.coefM = pd.DataFrame({'name': self.coefN,'coef': self.coef})
        self.selectCoef = self.coefM.loc[np.abs(self.coefM.coef) > threshold]
        self.selectFeatures = self.selectCoef.name
        

class RandomForest_Selection (SubsetSelection):

    def __init__(self, data, para ,response = 'target', n_tree = 5,  threshold = 0.5):
        from Time_Series.CrossValidation import grid_tune_parameter
        from sklearn.ensemble import RandomForestRegressor
        mdl = RandomForestRegressor(n_estimators = n_tree)
        n_feature = para
        super().__init__(mdl, data, response)
        self.para = grid_tune_parameter(mdl,data,response,[len(data)-1],n_feature,'n_features_').para
        self.para = int(self.para)
        mdl = RandomForestRegressor(n_estimators = n_tree, max_features= self.para)
        mdl.fit(self.datax, self.datay)
        self.threshold = threshold
        self.coef = self.mdl.feature_importances_
        self.coefN = self.datax.columns.tolist()
        self.coefM = pd.DataFrame({'name': self.coefN,'coef': self.coef})
        self.selectCoef = self.coefM.loc[np.abs(self.coefM.coef) > threshold]
        self.selectFeatures = self.selectCoef.name


class PCA_Selection (SubsetSelection):
    def __init__(self, mdl, data, response = 'target'):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        pca = PCA()
        super().__init__(mdl, data, response)
        trx = StandardScaler().fit_transform(X = data[0:len(self.datax)-1])
        testx = self.datax.tail(1)
        testy = self.datay.tail(1)
        pc = pca.fit(trx).components_
        pcDict = {}
        for i in range(1,len(pc)):
            invPC = pca.inverse_transform(pc[0:i])
            fitx = pd.DataFrame(columns = data.columns.tolist())
            for j in range(len(invPC)):
                tempdf = pd.DataFrame(invPC[j].reshape(1,49),columns = data.columns.tolist())
                fitx = fitx.append(tempdf, True)
            trainx = fitx.drop(response, axis = 1)
            trainy = fitx[response]
            mdl.fit(trainx,trainy)
            pcDict[i] = mdl.score(testx,testy)
        self.pcNum = max(pcDict,key = pcDict.get)
        self.PC = pc
        tempx = {}
        for i in range(self.pcNum):
            tempx['PC'+ str(i+1)] = pc[i]
        self.selectPC = pd.DataFrame(tempx)
        self.selectFeatures = self.selectPC.columns.tolist()
        self.threshold = self.pcNum
        self.coef = pd.DataFrame(columns = data.columns.tolist())
        for j in range(self.pcNum):
            tempdf = pd.DataFrame(pca.inverse_transform(self.PC[j]).reshape(1,49),columns = data.columns.tolist())
            self.coef = self.coef.append(tempdf,True)