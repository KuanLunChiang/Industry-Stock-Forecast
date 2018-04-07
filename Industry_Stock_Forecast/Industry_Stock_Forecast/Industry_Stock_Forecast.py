from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np

_data = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A3\Industry_Stock_Forecast\Industry_Stock_Forecast\Data\Data_2018.csv")
_data = _data.drop('Unnamed: 0', axis = 1)
_data.describe()
########### Class Attribute ###########################################
_colName = _data.columns.tolist()
_dataDict = {}
_responseVar = 'target'
_windowList = [1,2,3]
_paraList = [1,2,3]
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



############### PCA ##########################################
from sklearn.decomposition import PCA, KernelPCA
pca = PCA(whiten = True)

pca.fit(_dataDict['Agric'].drop(_responseVar, axis = 1))
pca.explained_variance_ratio_.cumsum()
pca.n_components_
pca.n_features_
pca.n_samples_
len(pca.components_)


################## LASSO ##########################################
from sklearn.linear_model import Lasso

mdl = Lasso(normalize = True)
tuneMdl = tcv.paralell_processing(mdl, _dataDict,_responseVar, _windowList, _paraList, 'alpha',_colName)  
