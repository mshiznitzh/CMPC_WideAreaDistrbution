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
    Outdoor_BreakerDF['Feeder_Protection'] = 'Non Sub'
    Outdoor_BreakerDF['FD_DEV_STATUS'] = np.nan

    # SUB IV
    df = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("SEL") & MODEL.str.contains("351")')

    df2 = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("SEL") & MODEL.str.contains("551")')

    df = df[df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    if 'Operating Needs Review' in df.DEV_STATUS.unique():
        dev = 'Operating Needs Review'
    elif 'Planned' in df.DEV_STATUS.unique():
        dev = 'Planned'
    elif 'Operating' in df.DEV_STATUS.unique():
        dev = 'Operating'

    Outdoor_BreakerDF['Feeder_Protection'] = np.where(Outdoor_BreakerDF['Feeder_Protection'].str.match('Non Sub') &
                                                       Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected),
                                                       'SUB IV',
                                                       Outdoor_BreakerDF['Feeder_Protection'])

    Outdoor_BreakerDF['FD_DEV_STATUS'] = np.where(
        Outdoor_BreakerDF['Feeder_Protection'].str.match('SUB IV') &
        Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), dev,
        Outdoor_BreakerDF['FD_DEV_STATUS'])

    # SUB II/III
    df = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("ABB") & MODEL.str.contains("DPU-2000R")')

    df2 = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("SEL") & MODEL.str.contains("501")')

    df = df[df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    if 'Operating Needs Review' in df.DEV_STATUS.unique():
        dev = 'Operating Needs Review'
    elif 'Planned' in df.DEV_STATUS.unique():
        dev = 'Planned'
    elif 'Operating' in df.DEV_STATUS.unique():
        dev = 'Operating'

    Outdoor_BreakerDF['Feeder_Protection'] = np.where(Outdoor_BreakerDF['Feeder_Protection'].str.match('Non Sub') &
                                                       Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected),
                                                       'SUB II/III',
                                                       Outdoor_BreakerDF['Feeder_Protection'])
    Outdoor_BreakerDF['FD_DEV_STATUS'] = np.where(
        Outdoor_BreakerDF['Feeder_Protection'].str.match('SUB II/III') &
        Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), dev,
        Outdoor_BreakerDF['FD_DEV_STATUS'])

    # DPU2000R_BE151_Standalone
    df = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("ABB") & MODEL.str.contains("DPU-2000R")')

    df2 = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("BASLER") & MODEL.str.contains("BE1-51")')

    df = df[df.Maximo_Asset_Protected.isin(df2.Maximo_Asset_Protected)]

    if 'Operating Needs Review' in df.DEV_STATUS.unique():
        dev = 'Operating Needs Review'
    elif 'Planned' in df.DEV_STATUS.unique():
        dev = 'Planned'
    elif 'Operating' in df.DEV_STATUS.unique():
        dev = 'Operating'

    Outdoor_BreakerDF['Feeder_Protection'] = np.where(Outdoor_BreakerDF['Feeder_Protection'].str.match('Non Sub') &
                                                      Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected),
                                                      'DPU2000R_BE151_Standalone',
                                                      Outdoor_BreakerDF['Feeder_Protection'])
    Outdoor_BreakerDF['FD_DEV_STATUS'] = np.where(
        Outdoor_BreakerDF['Feeder_Protection'].str.match('DPU2000R_BE151_Standalone') &
        Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), dev,
        Outdoor_BreakerDF['FD_DEV_STATUS'])

    # SUB I
    df = RelayDataDF.query(
        'PROT_TYPE.isin(["DISTRIBUTION FEEDER", "DISTRIBUTION FEEDER - UNDERFREQUENCY"]) & MFG.str.match("ABB") & MODEL.str.match("DPU")')

    df = df[df.Maximo_Asset_Protected.isin(df.Maximo_Asset_Protected)]

    if 'Operating Needs Review' in df.DEV_STATUS.unique():
        dev = 'Operating Needs Review'
    elif 'Planned' in df.DEV_STATUS.unique():
        dev = 'Planned'
    elif 'Operating' in df.DEV_STATUS.unique():
        dev = 'Operating'

    Outdoor_BreakerDF['Feeder_Protection'] = np.where(Outdoor_BreakerDF['Feeder_Protection'].str.match('Non Sub') &
        Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), 'SUB I',
        Outdoor_BreakerDF['Feeder_Protection'])

    Outdoor_BreakerDF['FD_DEV_STATUS'] = np.where(
        Outdoor_BreakerDF['Feeder_Protection'].str.match('SUB I') &
        Outdoor_BreakerDF['Maximo_Code'].isin(df.Maximo_Asset_Protected), dev,
        Outdoor_BreakerDF['FD_DEV_STATUS'])


    return Outdoor_BreakerDF