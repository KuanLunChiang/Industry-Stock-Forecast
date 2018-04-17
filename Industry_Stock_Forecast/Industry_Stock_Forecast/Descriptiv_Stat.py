from Time_Series import CrossValidation as tcv
from Time_Series import Report as rpt
import pandas as pd
import numpy as np
import seaborn as sns




_data = pd.read_csv(r".\Data\Data_2018.csv")
_data = _data.drop('Unnamed: 0', axis = 1)
_colName = _data.columns.tolist()
_data.describe()
_data.skew()
################ Correlation #################################
_corr = _data.corr()
_corr.to_csv(r'.\Output\Correlation_Matrix.csv')
sns.heatmap(_corr)
_avgCorr = {}
for i in _colName:
     _avgCorr[i]= _corr.stack().loc[i].mean()
_avgCorr = pd.Series(_avgCorr)
_avgCorr.nlargest(3)
_avgCorr.nsmallest(3)