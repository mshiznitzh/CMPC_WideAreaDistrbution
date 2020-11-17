import logging
import numpy as np

log = logging.getLogger(__name__)

def Fault_Reporting_Proiritization_df_cleanup(FRPdf):
    '''Clean up will rename the 9th coloumn to DOC_Fault_Reporting_Prioritization.  Function returns FRPdf'''
    log.info("Fault_Reporting_Proiritization_df_cleanup Starting")
    # logger.info("DataFrame has "+ FRPdf['SUBSTATION_NAME'].str.len() +" rows")
    FRPdf = FRPdf.rename(columns={FRPdf.columns[9]: 'DOC_Fault_Reporting_Prioritization'})
    FRPdf['DOC_Fault_Reporting_Prioritization'] = np.where(FRPdf['DOC_Fault_Reporting_Prioritization'].isnull(),
                                                           FRPdf['Feeder_Ranking'],
                                                           FRPdf['DOC_Fault_Reporting_Prioritization'])

    FRPdf['DOC_Fault_Reporting_Prioritization'] = np.where(FRPdf['DOC_Fault_Reporting_Prioritization'].isnull(),
                                                           FRPdf['PRIORITY:_HIGH,MEDIUM,LOW'],
                                                           FRPdf['DOC_Fault_Reporting_Prioritization'])

    FRPdf['DOC_Fault_Reporting_Prioritization'] = np.where(FRPdf['DOC_Fault_Reporting_Prioritization'].isnull(),
                                                           FRPdf['HIIGH-MEDIUM-LOW'],
                                                           FRPdf['DOC_Fault_Reporting_Prioritization'])

    FRPdf['DOC_Fault_Reporting_Prioritization'] = np.where(FRPdf['DOC_Fault_Reporting_Prioritization'].isnull(),
                                                           FRPdf['Priority'],
                                                           FRPdf['DOC_Fault_Reporting_Prioritization'])

    FRPdf['DOC_Fault_Reporting_Prioritization'] = np.where(FRPdf['FAULT_REPORTING'] == 'Y',
                                                           'Fault Reporting Enabled',
                                                           FRPdf['DOC_Fault_Reporting_Prioritization'])
    return FRPdf


def Fault_Reporting_Proiritization_df_create_data(FRPdf):
    FRPdf['Maximo_Code'] = FRPdf['FEEDER_ID'].str.slice(start=0, stop=5) + '-' + '0' + FRPdf['FEEDER_ID'].str.slice(
        start=5)
    FRPdf['Maximo_Code'] = FRPdf['Maximo_Code'].str.strip()

    return FRPdf