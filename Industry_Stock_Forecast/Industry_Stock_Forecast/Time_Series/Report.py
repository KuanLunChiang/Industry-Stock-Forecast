
import pandas as pd
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from matplotlib import pyplot as plt

class cum_sse_report (object):
    """
    This class will generate a report include cumulative differential SSE and cumulative out-of-sample R-squareed
    """
    def __init__(self, mdlError, benchError):
        resDf = pd.DataFrame({'RegError': mdlError, 'MeanError': benchError})
        resDf['cum_reg_error'] = resDf.RegError.cumsum()
        resDf['cum_mean_error'] = resDf.MeanError.cumsum()
        resDf['SSEDif'] =  resDf.cum_mean_error - resDf.cum_reg_error
        resDf['oosrsquare'] = 1- (resDf.cum_reg_error / resDf.cum_mean_error)
        self.reportDF = resDf



class plot_differential_report ():
    """
    An interface for ploting differential report
    """

    def __init__(self, colName, data, para ,row = 3, col = 3, supT = 'Untitled'):
        pltind = row*100+col*10+1
        plt.figure(figsize=(15,15))
        for i in colName:
            plt.subplot(pltind)
            data[i][para].plot()
            plt.title(i)
            plt.suptitle(supT, fontsize = 15, y = 0.92)
            pltind +=1
        return plt.show()


class outPutReport ():
    
    def __init__(self, mdl ,name):
        coefDF = mdl.coefList
        errorDF = mdl.errorList
        winParaDF = mdl.report_tuned
        self.json_output(coefDF,'.\\Output\\Coefficient\\'+name+'_coefficient'+'.txt')
        self.json_output(errorDF,'.\\Output\\Error List\\'+name+'_errorList'+'.txt')
        winParaDF.to_csv('.\\Output\\Window and Parameter\\'+name+'_winPara'+'.csv')


    def json_output (self, data, output):
        import json
        with open(output, 'w') as outfile:
            json.dump(data,outfile)