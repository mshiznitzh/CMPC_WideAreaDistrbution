import pandas as pd
import numpy as np


def Add_Feeder_Protection_on_Bank(PowerTransformerDF, Outdoor_BreakerDF):
    Outdoor_BreakerDF_sorted = Outdoor_BreakerDF.sort_values(by=['Feeder_Protection'])
    Outdoor_BreakerDF_sorted = Outdoor_BreakerDF_sorted.drop_duplicates(subset=['Associated_XFMR'], keep=("first"))

    PowerTransformerDF = pd.merge(PowerTransformerDF,
                                  Outdoor_BreakerDF_sorted[['Associated_XFMR', 'Feeder_Protection']],
                                  left_on='Maximo_Code', right_on='Associated_XFMR', how='left')

    return PowerTransformerDF

def Add_High_Side_Interrupter_PowerTransformerDF(PowerTransformerDF, High_Side_Protection_DF):

    PowerTransformerDF = pd.merge(PowerTransformerDF, High_Side_Protection_DF[['Maximo_Code', 'High_Side_Interrupter']], how= 'left', on='Maximo_Code')

    PowerTransformerDF['High_Side_Interrupter'] = np.where(PowerTransformerDF['High_Side_Interrupter'] == 'B' ,
                                                           'Breaker' ,
                                                           PowerTransformerDF['High_Side_Interrupter'])

    PowerTransformerDF['High_Side_Interrupter'] = np.where(PowerTransformerDF['High_Side_Interrupter'] == 'CS',
                                                           'Circuit Switcher',
                                                           PowerTransformerDF['High_Side_Interrupter'])

    PowerTransformerDF['High_Side_Interrupter'] = np.where(PowerTransformerDF['High_Side_Interrupter'] == 'FB',
                                                           'Flash Bus',
                                                           PowerTransformerDF['High_Side_Interrupter'])

    PowerTransformerDF['High_Side_Interrupter'] = np.where(PowerTransformerDF['High_Side_Interrupter'] == 'GS',
                                                           'Ground Switch',
                                                           PowerTransformerDF['High_Side_Interrupter'])

    return PowerTransformerDF


def add_Xfmer_Diff_Protection_PowerTransformerDF(RelayDataDF, PowerTransformerDF):
    RelayDataDF['Xfmer_Diff_Protection'] = 'Non Sub'
    RelayDataDF['Xfmer_Diff_Protection'] = np.where(RelayDataDF['MODEL'].str.match('DPU') &
                                                    RelayDataDF['MFG'].str.contains('ABB'), 'SUB 1',
                                                    RelayDataDF['Xfmer_Diff_Protection'])

    RelayDataDF['Xfmer_Diff_Protection'] = np.where(RelayDataDF['MODEL'].str.contains('STD') &
                                                    RelayDataDF['MFG'].str.match('GE'), 'SUB 2',
                                                    RelayDataDF['Xfmer_Diff_Protection'])

    RelayDataDF['Xfmer_Diff_Protection'] = np.where(RelayDataDF['MODEL'].str.match('SEL-587') &
                                                    RelayDataDF['MFG'].str.contains('SEL'), 'SUB 3',
                                                    RelayDataDF['Xfmer_Diff_Protection'])

    RelayDataDF['Xfmer_Diff_Protection'] = np.where(RelayDataDF['MODEL'].str.contains('387') &
                                                    RelayDataDF['MFG'].str.match('SEL'), 'SUB 4',
                                                    RelayDataDF['Xfmer_Diff_Protection'])

    list = ['SUB 1', 'SUB 2', 'SUB 3', 'SUB 4', 'Non Sub']

    df = RelayDataDF[RelayDataDF['Xfmer_Diff_Protection'].isin(list)]

    PowerTransformerDF = pd.merge(PowerTransformerDF,
                                  df[['Maximo_Asset_Protected', 'Xfmer_Diff_Protection']],
                                  left_on='Maximo_Code', right_on='Maximo_Asset_Protected', how='left')

    PowerTransformerDF = PowerTransformerDF.sort_values(by=['Xfmer_Diff_Protection']).drop_duplicates(
        subset=['Maximo_Code'], keep=("last"))

    return PowerTransformerDF


def add_Xfmer2_Diff_Protection_PowerTransformerDF(RelayDataDF, PowerTransformerDF):
    PowerTransformerDF['Xfmer_Diff_Protection'] = 'Non Sub'

    # SUB IV
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("387")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 2)
    df2 = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("2100")')

    df = df[df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'SUB IV',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    # SUB III
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("587")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 1)

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'SUB III',
        PowerTransformerDF['Xfmer_Diff_Protection'])


    # Dual 587 Retrofit
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("587")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 2)
    # df2 = RelayDataDF.query(
    #   'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("501")')

    # retrodf = dual587df[dual587df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'Dual 587 Retrofit',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    # Dual 387 Retrofit
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("387")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 2)

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'Dual 387 Retrofit',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    # SUB II
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("GE") & MODEL.str.contains("STD")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 3)

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'SUB II',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("GE") & MODEL.str.contains("STD")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) != 3)

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'Have District Verify',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    # SUB I
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("GE") & MODEL.str.contains("PJC")')

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['Xfmer_Diff_Protection'].str.match('Non Sub') &
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'SUB I',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    #Fused
    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['IsFused'],
        'Fused',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    return PowerTransformerDF
