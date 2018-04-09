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
    def __init__(self, data, response = 'target', n_tree = 10, n_feature = np.arange(1,len(data),1), threshold = 0.5):
        from Time_Series.CrossValidation import grid_tune_parameter
        from sklearn.ensemble import RandomForestRegressor
        mdl = RandomForestRegressor(n_estimators = n_tree)
        super().__init__(mdl, data, response)
        self.featureNum = grid_tune_parameter(mdl,data,response,[len(data)-1],n_feature,'n_features_').para
        mdl = RandomForestRegressor(n_estimators = n_tree, max_features = self.featureNum)
        mdl.fit(self.datax, self.datay)
        self.threshold = threshold
        self.coef = self.mdl.feature_importances_
        self.coefN = self.datax.columns.tolist()
        self.coefM = pd.DataFrame({'name': self.coefN,'coef': self.coef})
        self.selectCoef = self.coefM.loc[np.abs(self.coefM.coef) > threshold]
        self.selectFeatures = self.selectCoef.name




class PCA_Selection (SubsetSelection):
    def __init__(self, mdl, data, response = 'target'):
        return super().__init__(mdl, data, response)