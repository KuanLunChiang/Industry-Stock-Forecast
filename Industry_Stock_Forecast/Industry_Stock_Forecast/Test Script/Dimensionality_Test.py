import pandas as pd
import numpy as np


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


################### LASSO selection ########################################################
from sklearn.linear_model import Lasso
from Dimensionality_Reduction.SubsetSelection import *
from sklearn.ensemble import RandomForestRegressor

train = _trainDict['Food']
datax = train.drop('target',axis = 1)
datay = train['target']
select = Lasso_Selection(train)
select.coefM
select.selectCoef
select.alpha
select.coefM

mdl = RandomForestRegressor(max_features = int(10.0))
mdl.fit(datax,datay)

a = [1,2,3]
a[0:len(a)-1]


from Time_Series.CrossValidation import grid_tune_parameter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca = PCA()
datax = train.drop('target',axis = 1)
datay = train['target']
trx = StandardScaler().fit_transform(X = datax[0:len(datax)-1])
pc = pca.fit(trx).components_
pcDict = {}
for i in range(1,len(pc)):
    trainx = pc[0:i]
    trainy = datay[0:i]
    testx = pc[len(pc)-1]
    testy = datay[len(datay)-1]
    mdl.fit(pca.inverse_transform(trainx),trainy)
    pcDict[i] = mdl.score(testx,testy)

from sklearn.linear_model import LinearRegression
mdl = LinearRegression()
i = 1
trainx = pc[0:i]
trainy = datay[0:i]
testx = pc[len(pc)-1]
testy = datay[len(datay)-1]
mdl.fit(pca.inverse_transform(trainx),trainy)
mdl.predict(testx)
pca.inverse_transform(trainx).shape
pcDict[i] = mdl.score(train.drop(_responseVar,axis = 1).tail(1),testy)