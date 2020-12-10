import numpy as np
import pandas as pd

def add_Relay_Outdoor_BreakerDF(RelayDataDF, Outdoor_BreakerDF):
    RelayDataDF['Feeder_Protection'] = 'Non Sub'
    RelayDataDF['Feeder_Protection'] = np.where(RelayDataDF['MODEL'].str.match('DPU') &
                                                RelayDataDF['MFG'].str.contains('ABB'), 'SUB 1',
                                                RelayDataDF['Feeder_Protection'])

    RelayDataDF['Feeder_Protection'] = np.where(RelayDataDF['MODEL'].str.match('DPU-2000R') &
                                                RelayDataDF['MFG'].str.contains('ABB'), 'SUB 2/3',
                                                RelayDataDF['Feeder_Protection'])

    RelayDataDF['Feeder_Protection'] = np.where(RelayDataDF['MODEL'].str.contains('351') &
                                                RelayDataDF['MFG'].str.match('SEL'), 'SUB 4',
                                                RelayDataDF['Feeder_Protection'])

    RelayDataDF['Feeder_Protection'] = np.where(RelayDataDF['MODEL'].str.contains('551') &
                                                RelayDataDF['MFG'].str.match('SEL'), 'SUB 4',
                                                RelayDataDF['Feeder_Protection'])

    list = ['SUB 1', 'SUB 2/3', 'SUB 4', 'Non Sub']

    df = RelayDataDF[RelayDataDF['Feeder_Protection'].isin(list)]

    Outdoor_BreakerDF = pd.merge(Outdoor_BreakerDF,
                                 df[['Maximo_Asset_Protected', 'Feeder_Protection']],
                                 left_on='Maximo_Code', right_on='Maximo_Asset_Protected', how='left')

    Outdoor_BreakerDF = Outdoor_BreakerDF.sort_values(by=['Feeder_Protection']).drop_duplicates(subset=['Maximo_Code'], keep=("last"))

    return Outdoor_BreakerDF


def add_Relay2_Outdoor_BreakerDF(RelayDataDF, Outdoor_BreakerDF):
    RelayDataDF['Feeder_Protection2'] = 'Non Sub'

    #SUB IV
    df = RelayDataDF.query(
        'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("587")')

    df = df.groupby('Maximo_Asset_Protected').filter(lambda x: len(x) == 2)
    # df2 = RelayDataDF.query(
    #   'PROT_TYPE.str.match("DISTRIBUTION TRANSFORMER") & MFG.str.match("SEL") & MODEL.str.contains("501")')

    # retrodf = dual587df[dual587df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    PowerTransformerDF['Xfmer_Diff_Protection'] = np.where(
        PowerTransformerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'Dual 587 Retrofit',
        PowerTransformerDF['Xfmer_Diff_Protection'])

    return Outdoor_BreakerDF