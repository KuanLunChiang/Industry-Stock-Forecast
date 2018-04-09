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
        return None

class RandomForest_Selection (SubsetSelection):

    def __init__(self, data, response = 'target', n_tree = 10,  threshold = 0.5):
        from Time_Series.CrossValidation import grid_tune_parameter
        from sklearn.ensemble import RandomForestRegressor
        mdl = RandomForestRegressor(n_estimators = n_tree)
        n_feature = np.arange(1,len(data)-1,1)
        super().__init__(mdl, data, response)
        self.para = grid_tune_parameter(mdl,data,response,[len(data)-1],n_feature,'n_features_').para
        self.para = int(self.para)
        mdl = RandomForestRegressor(n_estimators = n_tree, max_features = self.para)
        mdl.fit(self.datax, self.datay)
        self.threshold = threshold
        self.coef = self.mdl.feature_importances_
        self.coefN = self.datax.columns.tolist()
        self.coefM = pd.DataFrame({'name': self.coefN,'coef': self.coef})
        self.selectCoef = self.coefM.loc[np.abs(self.coefM.coef) > threshold]
        self.selectFeatures = self.selectCoef.name





### Need Debug, not finished
class PCA_Selection (SubsetSelection):
    def __init__(self, mdl, data, response = 'target'):
        from Time_Series.CrossValidation import grid_tune_parameter
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        pca = PCA()
        super().__init__(mdl, data, response)
        trx = StandardScaler().fit_transform(X = self.datax[0:len(self.datax)-1])
        pc = pca.fit(trx).components_
        pcDict = {}
        for i in range(1,len(pc)):
            trainx = pc[0:i]
            trainy = self.datay[0:i]
            testx = pc[len(pc)-1]
            testy = self.datay[len(self.datay)-1]
            trainx = pca.inverse_transform(trainx)
            mdl.fit(pca.inverse_transform(trainx),trainy)
            pcDict[i] = mdl.score(testx,testy)
        self.pcNum = max(pcDict,key = pcDict.get)
        self.PC = pc
        self.selecteFeatures = pc[0:self.pcNum]