import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from Dimensionality_Reduction.SubsetSelection import *
from sklearn.ensemble import RandomForestRegressor

############### Initialization ###################################################
_data = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A3\Industry-Stock-Forecast\Industry_Stock_Forecast\Industry_Stock_Forecast\Data\Data_2018.csv")
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
    _trainDict[i] = _dataDict[i].iloc[0:800]
    _testDict[i] = _dataDict[i].drop(_trainDict[i].index)

train = _trainDict['Food']
##################################################################################

class Test_test_dim(unittest.TestCase):
    def test_randomForest(self):
        select = RandomForest_Selection(train)
        print(select.coefM)
        print(select.selectCoef)


    def test_lasso(self):
        select = Lasso_Selection(train)
        print(select.coefM)
        print(select.selectCoef)

    def test_PCA (self):
        from sklearn.linear_model import LinearRegression
        mdl = LinearRegression()
        select = PCA_Selection(mdl,train)

if __name__ == '__main__':
    unittest.main()
