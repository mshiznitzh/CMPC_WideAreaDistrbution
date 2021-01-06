# This is a sample Python script.

# !/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "MiKe Howard"
__version__ = "0.1.0"
__license__ = "MIT"

import logging
from logzero import logger
import pandas as pd
import glob
import os
import datetime as dt
import numpy as np
from multiprocessing import Pool, cpu_count
import FRP
from tqdm import tqdm
from itertools import combinations
from itertools import zip_longest
import smart_excel_to_pandas.smart_excel_to_pandas
import PowerTransformer
import Outdoor_Breaker

# OS Functions

def filesearch(word=""):
    """Returns a list with all files with the word/extension in it"""
    logger.info('Starting filesearch')
    file = []
    for f in glob.glob("*"):
        if word[0] == ".":
            if f.endswith(word):
                file.append(f)

        elif word in f:
            file.append(f)
            # return file
    logger.debug(file)
    return file


def Change_Working_Path(path):
    """Check if New path exists if the path exist the working path will be changed else will print an error message"""
    if os.path.exists(path):
        # Change the current working Directory
        try:
            os.chdir(path)  # Change the working directory
        except OSError:
            logger.error("Can't change the Current Working Directory", exc_info=True)
    else:
        print("Can't change the Current Working Directory because this path doesn't exits")


# Pandas Functions
def Excel_to_Pandas(filename, check_update=False, SheetName=None):
    """Imports excel file to pandas returns filename and df"""
    logger.info('importing file ' + filename)
    df = []
    if check_update == True:
        timestamp = dt.datetime.fromtimestamp(Path(filename).stat().st_mtime)
        if dt.datetime.today().date() != timestamp.date():
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(title=' '.join(['Select file for', filename]))

    try:
        df = pd.read_excel(filename, SheetName)
        df = pd.concat(df, axis=0, ignore_index=True)
    except:
        logger.error("Error importing file " + filename, exc_info=True)

    df = Cleanup_Dataframe(df)
    logger.info(df.info(verbose=True))
    return (filename, df)


def Cleanup_Dataframe(df):
    logger.info('Starting Cleanup_Dataframe')
    logger.debug(df.info(verbose=False))
    # Remove whitespace on both ends of column headers
    df.columns = df.columns.str.strip()

    # Replace whitespace in column header with _
    df.columns = df.columns.str.replace(' ', '_')

    return df


def station_df_cleanup(StationDF, Metalclad_Switchgear_DF):
    # Filters
    StationDF = StationDF[StationDF['Ownership_Name'].str.contains('ONCOR ELECTRIC DELIVERY')]
    StationDF = StationDF[StationDF['STATION_CLASS'].str.contains('SUBSTATION')]
    StationDF = StationDF[~StationDF['Station_Name'].isin(Metalclad_Switchgear_DF['Station_Name'])]

    # Convert Datatypes
    StationDF['IN_SERVICE_DATE'] = pd.to_datetime(StationDF['IN_SERVICE_DATE'], errors='raise')

    return StationDF


def transformer_df_cleanup(TransformerDF):
    TransformerDF = TransformerDF[TransformerDF['XFMR_SERVICE'].str.contains('POWER')]

    return TransformerDF


def transformer_df_create_data(PowerTransformerDF, Transformer_RiskDF, Summer_LoadDF, Winter_LoadDF, AIStationDF):
    PowerTransformerDF = PowerTransformerDF[PowerTransformerDF['Station_Name'].isin(list(AIStationDF['Station_Name']))]
    PowerTransformerDF['Age'] = (dt.date.today() - PowerTransformerDF['Manufacture_Date'].dt.date) / 365
    Transformer_RiskDF = Transformer_RiskDF.rename(columns={"Asset": "Maximo_Code"})
    PowerTransformerDF = pd.merge(PowerTransformerDF, Transformer_RiskDF[['Maximo_Code', 'Risk_Index_(Normalized)']],
                                  on='Maximo_Code', how='left')

    PowerTransformerDF = pd.merge(PowerTransformerDF, Summer_LoadDF[['Maximo_Code', 'Projected_Summer_Load_2020',
                                                                     'Projected_Summer_Load_2021',
                                                                     'Projected_Summer_Load_2022',
                                                                     'Projected_Summer_Load_2023',
                                                                     'Projected_Summer_Load_2024',
                                                                     'Projected_Summer_Load_2025']], on='Maximo_Code',
                                  how='left')

    PowerTransformerDF = pd.merge(PowerTransformerDF, Winter_LoadDF[['Maximo_Code', 'Projected_Winter_Load_2020',
                                                                     'Projected_Winter_Load_2021',
                                                                     'Projected_Winter_Load_2022',
                                                                     'Projected_Winter_Load_2023',
                                                                     'Projected_Winter_Load_2024',
                                                                     'Projected_Winter_Load_2025']], on='Maximo_Code',
                                  how='left')

    for item in ['Projected_Summer_Load_2021', 'Projected_Summer_Load_2022', 'Projected_Summer_Load_2023',
                 'Projected_Summer_Load_2024', 'Projected_Summer_Load_2025', 'Projected_Winter_Load_2021',
                 'Projected_Winter_Load_2022', 'Projected_Winter_Load_2023',
                 'Projected_Winter_Load_2024', 'Projected_Winter_Load_2025']:
        PowerTransformerDF[item] = np.where(PowerTransformerDF['NUM_PH'] == 1, PowerTransformerDF[item] / 3,
                                            PowerTransformerDF[item])

    PowerTransformerDF['Max_Projected_Summer_Load'] = PowerTransformerDF[['Projected_Summer_Load_2021',
                                                                          'Projected_Summer_Load_2022',
                                                                          'Projected_Summer_Load_2023',
                                                                          'Projected_Summer_Load_2024',
                                                                          'Projected_Summer_Load_2025']].max(axis=1)

    PowerTransformerDF['Max_Projected_Winter_Load'] = PowerTransformerDF[['Projected_Winter_Load_2021',
                                                                          'Projected_Winter_Load_2022',
                                                                          'Projected_Winter_Load_2023',
                                                                          'Projected_Winter_Load_2024',
                                                                          'Projected_Winter_Load_2025']].max(axis=1)

    PowerTransformerDF['Max_MVA_Exceeded'] = False
    PowerTransformerDF['Max_MVA_Exceeded'] = np.where(
        (PowerTransformerDF['Max_Projected_Summer_Load'] > PowerTransformerDF['MAXIMUM_MVA']) |
        (PowerTransformerDF['Max_Projected_Winter_Load'] > PowerTransformerDF['MAXIMUM_MVA']), True,
        PowerTransformerDF['Max_MVA_Exceeded'])

    return PowerTransformerDF


def add_Risk_to_Stationdf(StationDF, PowerTransformerDF):
    maxDF = PowerTransformerDF.groupby(['Station_Name'], as_index=False).agg(
        Max_Risk_Index_at_Station=('Risk_Index_(Normalized)', pd.Series.max))
    StationDF = pd.merge(StationDF, maxDF, on='Station_Name', how='left')
    StationDF.drop_duplicates(inplace=True)
    return StationDF


def add_MVA_Exceeded_Stationdf(StationDF, PowerTransformerDF):
    Exceededdf = PowerTransformerDF[PowerTransformerDF['Max_MVA_Exceeded']]
    StationDF = pd.merge(StationDF, Exceededdf[['Station_Name', 'Max_MVA_Exceeded']], how='left', on='Station_Name')
    StationDF['Max_MVA_Exceeded'] = np.where(StationDF['Max_MVA_Exceeded'].isnull(), False,
                                             StationDF['Max_MVA_Exceeded'])
    StationDF.drop_duplicates(inplace=True)
    return StationDF


def station_df_create_data(StationDF, PowerTransformerDF, Outdoor_BreakerDF):
    StationDF['Age'] = (dt.date.today() - StationDF['IN_SERVICE_DATE'].dt.date) / 365

    # Get a count of the number of XMFR at a station
    countDF = PowerTransformerDF.groupby(['Station_Name'], as_index=False).agg(
        XFMER_Count=('Asset_Number', pd.Series.count))
    StationDF = pd.merge(StationDF, countDF, on='Station_Name', how='left')

    # Build indication for 1 PH Stations

    Single_Phase_Stations_DF = PowerTransformerDF[PowerTransformerDF['NUM_PH'] == 1]
    Single_Phase_Stations_DF['Single_Phase_Station'] = True
    StationDF = pd.merge(StationDF, Single_Phase_Stations_DF[['Station_Name', 'Single_Phase_Station']],
                         on='Station_Name', how='left')
    StationDF['Single_Phase_Station'] = np.where(StationDF['Single_Phase_Station'].isnull(), False,
                                                 StationDF['Single_Phase_Station'])

    # Get a Avg Age of breakers at station
    Outdoor_BreakerDF['Age'] = (dt.date.today() - Outdoor_BreakerDF['Manufacture_Date'].dt.date) / 365
    Outdoor_FeedersDF = Outdoor_BreakerDF[Outdoor_BreakerDF['BKR_SERVICE'].str.match('FEEDER')]
    meanDF = Outdoor_FeedersDF.groupby(['Station_Name']).agg(Mean_Feeder_Age=('Age', pd.Series.mean))
    StationDF = pd.merge(StationDF, meanDF, on='Station_Name', how='left')

    StationDF.drop_duplicates(inplace=True)
    return StationDF


def breaker_df_cleanup(BreakerDF):
    BreakerDF = BreakerDF[BreakerDF['BKR_CLASS'].str.contains('OUTDOOR')]
    BreakerDF['Manufacture_Date'] = pd.to_datetime(BreakerDF['Manufacture_Date'], errors='raise')
    BreakerDF['SELF_CONTAINED'].replace('SELF-CONTAINED', True, inplace=True)
    BreakerDF['SELF_CONTAINED'].replace('Y', True, inplace=True)
    BreakerDF['SELF_CONTAINED'].replace('NON SELF-CONTAINED', False, inplace=True)
    BreakerDF['SELF_CONTAINED'].replace('N', False, inplace=True)

    return BreakerDF


def breaker_df_create_data(BreakerDF, PowerTransformerDF, Fault_Reporting_ProiritizationDF):
    BreakerDF = pd.merge(BreakerDF,
                         Fault_Reporting_ProiritizationDF[['Maximo_Code', 'DOC_Fault_Reporting_Prioritization']],
                         how='left', on='Maximo_Code')

    stations_with_Single_BankDF = PowerTransformerDF[
        PowerTransformerDF['Station_Name'].map(PowerTransformerDF['Station_Name'].value_counts()) == 1]

    Single_Bank = list(stations_with_Single_BankDF['Station_Name'])

    return BreakerDF


def relay_df_cleanup(relaydf):
    relaydf = relaydf[~relaydf['DEV_STATUS'].str.match('Planned')]
    relaydf = relaydf.reset_index(drop=True)
    return relaydf


def relay_df_create_data(relaydf, PowerTransformerDF, Outdoor_BreakerDF):
    relaydf['Maximo_Asset_Protected_Device_Type'] = np.nan
    relaydf['Maximo_Asset_Protected'] = np.nan

    relaydf['Maximo_Asset_Protected_Station'] = relaydf['LOCATION'].str.slice(start=0, stop=5)
    relaydf['Maximo_Asset_Protected_Device_Type'] = np.where(relaydf['PROT_TYPE'].str.match('DISTRIBUTION FEEDER'),
                                                             'BKR',
                                                             relaydf['Maximo_Asset_Protected_Device_Type'])

    relaydf['Maximo_Asset_Protected_Device_Type'] = np.where(relaydf['PROT_TYPE'].str.match('DISTRIBUTION TRANSFORMER'),
                                                             'TRF',
                                                             relaydf['Maximo_Asset_Protected_Device_Type'])

    relaydf['Maximo_Asset_Protected_Device_Num'] = relaydf['LOCATION'].str.extract('(\d+$)').astype(str)

    relaydf = parallelize_dataframe(relaydf, old_funtion, PowerTransformerDF, Outdoor_BreakerDF)
    #relaydf = old_funtion(relaydf, PowerTransformerDF, Outdoor_BreakerDF)

    return relaydf

def parallelize_dataframe(df, func, df1, df2):
    cpu_used = cpu_count()-1
    df_split = np.array_split(df, cpu_used)
    arr = [(df_split[x], df1, df2) for x in range(len(df_split))]


    with Pool(cpu_used) as p:

        df = pd.concat(p.starmap(func, arr))
        p.close()
        p.join()

    df = df.reset_index(drop=True)
    return df

def old_funtion(relaydf, PowerTransformerDF, Outdoor_BreakerDF):

    relaydf = relaydf.reset_index(drop=True)
    #for index in range(relaydf.shape[0]):
    for index in tqdm(range(relaydf.shape[0]), desc='Looping over relay Dataframe'):
        if relaydf['Maximo_Asset_Protected_Device_Type'][index] == 'TRF':
            df = PowerTransformerDF[(PowerTransformerDF['Maximo_Code'].str.contains(relaydf['Maximo_Asset_Protected_Station'][index])) &
                                    (PowerTransformerDF['Maximo_Code'].str.contains(str(relaydf['Maximo_Asset_Protected_Device_Num'][index])))
                                    ]

        elif relaydf['Maximo_Asset_Protected_Device_Type'][index] == 'BKR':
            df = Outdoor_BreakerDF[(Outdoor_BreakerDF['Maximo_Code'].str.contains(relaydf['Maximo_Asset_Protected_Station'][index])) &
                                   (Outdoor_BreakerDF['Maximo_Code'].str.contains(relaydf['Maximo_Asset_Protected_Device_Num'][index]))
                                   ]

        if df.shape[0] >> 0:
            try:
                relaydf['Maximo_Asset_Protected'].iloc[index] = df['Maximo_Code'].iloc[0]
            except:
               print(df['Maximo_Code'])
        else:
            relaydf['Maximo_Asset_Protected'][index] = np.nan
    return relaydf

def summer_load_df_cleanup(Summer_LoadDF):
    # One offs
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DEALEY STREET #1', 'DEALEY #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DEALEY STREET #2', 'DEALEY #2')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('SOUTH CLIFF PUMP ST #1',
                                                                            'SOUTH CLIFF PUMP')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DALLAS WEST #1', 'WEST DALLAS #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DALLAS WEST #2', 'WEST DALLAS #2')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BOBTOWN ROAD #1', 'BOBTOWN #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BOBTOWN ROAD #2', 'BOBTOWN #2')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DALROCK ROAD #1', 'DALROCK')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('Dalrock Road #2 Bank', 'DALROCK')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('DALROCK ROAD #4', 'DALROCK')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('Duck Cove #1(BANK)', 'DUCK COVE #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('MESQUITE DUCK CREEK WWTP #1',
                                                                            'DUCK CREEK #1')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('ELM GROVE ROAD', 'ELM GROVE #1')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('FATE #1', 'FATE SUB')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('JIM MILLER ROAD PUMP STATION #1',
                                                                            'JIM MILLER PUMP #1')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BIG SPRING AIRPARK #1',
                                                                            'BIG SPRING AIR PARK #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BIG SPRING AIRPARK #2',
                                                                            'BIG SPRING AIR PARK #2')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BRECKENRIDGE #1', 'BRECKENRIDGE #1')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('BRECKENRIDGE #2', 'BRECKENRIDGE #2')
    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('Breckenridge #3 Bank', 'BRECKENRIDGE #3')

    Summer_LoadDF['Transformer'] = Summer_LoadDF['Transformer'].str.replace('CHICO CITY SERVICES #1',
                                                                            'CHICO CITIES SERVICE #1')

    Summer_LoadDF['Transformer_alpha'] = Summer_LoadDF['Transformer'].str.replace('[^a-zA-Z\s]', '')
    Summer_LoadDF['Transformer_alpha'] = Summer_LoadDF['Transformer_alpha'].str.lstrip()
    Summer_LoadDF['Transformer_alpha'] = Summer_LoadDF['Transformer_alpha'].str.rstrip()
    Summer_LoadDF['Transformer_numeric'] = Summer_LoadDF['Transformer'].str.replace('[^0-9]', '')

    return Summer_LoadDF


def summer_load_df_create_data(Summer_LoadDF, AIStationDF):
    Summer_LoadDF = pd.merge(Summer_LoadDF, AIStationDF[['Station_Name', 'Maximo_Code']], left_on='Transformer_alpha',
                             right_on='Station_Name', how='left')

    Summer_LoadDF['Maximo_Code'] = Summer_LoadDF['Maximo_Code'] + '-TRF0' + Summer_LoadDF['Transformer_numeric']

    return Summer_LoadDF













def Add_Associated_XMR_Details(Outdoor_BreakerDF, Associated_Breaker_DetailsDF):
    Associated_Breaker_DetailsDF = Associated_Breaker_DetailsDF.rename(
        columns={"Maximo_Code": "Associated_XFMR", "Associated_Breaker_Maximo_Code": "Maximo_Code"})
    Outdoor_BreakerDF = pd.merge(Outdoor_BreakerDF, Associated_Breaker_DetailsDF[['Maximo_Code', 'Associated_XFMR']],
                                 on='Maximo_Code', how='left')

    return Outdoor_BreakerDF


def Add_fused_Bank_to_PowerTransformerDF(PowerTransformerDF, RelayDataDF):
    PowerTransformerDF['IsFused'] = True
    PowerTransformerDF['IsFused'] = np.where(
        PowerTransformerDF['Maximo_Code'].isin(RelayDataDF['Maximo_Asset_Protected']),
        False, PowerTransformerDF['IsFused'])

    return PowerTransformerDF


def Add_Fused_Bank_to_Stationdf(AIStationDF, PowerTransformerDF):
    AIStationDF['Has_Fused_Bank'] = False
    fusedDF = PowerTransformerDF[PowerTransformerDF['IsFused'] == True]
    AIStationDF['Has_Fused_Bank'] = np.where(AIStationDF['Station_Name'].isin(fusedDF['Station_Name']), True
                                             , AIStationDF['Has_Fused_Bank'])

    return AIStationDF





def Add_Feeder_Protection_on_Station(PowerTransformerDF, AIStationDF):
    PowerTransformerDF_sorted = PowerTransformerDF.sort_values(by=['Feeder_Protection'])
    PowerTransformerDF_sorted = PowerTransformerDF_sorted.drop_duplicates(subset=['Maximo_Code'], keep=("first"))

    AIStationDF = pd.merge(AIStationDF, PowerTransformerDF_sorted[['Station_Name', 'Feeder_Protection']],
                           left_on='Station_Name', right_on='Station_Name', how='left')

    AIStationDF = AIStationDF.drop_duplicates(subset=['Maximo_Code'], keep=("first"))

    return AIStationDF

def Add_Xfmer_Diff_Protection_on_Station(PowerTransformerDF, AIStationDF):
    PowerTransformerDF_sorted = PowerTransformerDF.sort_values(by=['Xfmer_Diff_Protection'])
    PowerTransformerDF_sorted = PowerTransformerDF_sorted.drop_duplicates(subset=['Maximo_Code'], keep=("first"))

    AIStationDF = pd.merge(AIStationDF, PowerTransformerDF_sorted[['Station_Name', 'Xfmer_Diff_Protection']],
                           left_on='Station_Name', right_on='Station_Name', how='left')

    AIStationDF = AIStationDF.drop_duplicates(subset=['Maximo_Code'], keep=("first"))

    return AIStationDF


def Add_Bus_Tie_at_Station(AIStationDF, Outdoor_BreakerDF):
    # Get a count of the number of XMFR at a station
    bustiedf = Outdoor_BreakerDF[Outdoor_BreakerDF['BKR_SERVICE'].str.match('BUS-TIE/BYPASS')]
    countDF = bustiedf.groupby(['Station_Name'], as_index=False).agg(
        BUS_TIE_Count=('Asset_Number', pd.Series.count))
    StationDF = pd.merge(AIStationDF, countDF[['Station_Name', 'BUS_TIE_Count']], on='Station_Name', how='left')

    StationDF['Bus_Equal_XFMER'] = False
    StationDF['Bus_Equal_XFMER'] = np.where(StationDF['BUS_TIE_Count'].gt(StationDF['XFMER_Count']/2),
                                            True,
                                            StationDF['Bus_Equal_XFMER'])


    return StationDF
def CS_df_cleanup(df):
    return df

def Add_CS_count_on_Station(Circuit_Switcher_df, AIStationDF):
    # Get a count of the number of XMFR at a station
    FIDdf = Circuit_Switcher_df[Circuit_Switcher_df['CSW_SERVICE'].str.match('TRANSFORMER', na=False)]
    countDF = FIDdf.groupby(['Station_Name'], as_index=False).agg(
        FID_Count=('Asset_Number', pd.Series.count))
    StationDF = pd.merge(AIStationDF, countDF[['Station_Name', 'FID_Count']], on='Station_Name', how='left')
    StationDF['FID_Count'] = np.where( StationDF['FID_Count'].isnull(), 0, StationDF['FID_Count'])
    return StationDF

def Add_FID_count_equal_XFMER_count(AIStationDF):
    AIStationDF['FIDequalXFMER'] = False
    AIStationDF['FIDequalXFMER'] = np.where(AIStationDF['XFMER_Count'] == AIStationDF['FID_Count'], True,  AIStationDF['FIDequalXFMER'])

    return AIStationDF

def Suggested_Approach_Station(AIStationDF):
    AIStationDF['Suggested_Approach_Station'] = np.nan

    # Verify Data

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Xfmer_Diff_Protection'].isnull()) &
                                                         (AIStationDF['Feeder_Protection'].isnull()),
                                                         'Verify that station has no Transformer',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         ~AIStationDF['Single_Phase_Station'] &
                                                         (AIStationDF['Feeder_Protection'].isnull()) &
                                                         (~AIStationDF['Mean_Feeder_Age'].isnull()),
                                                         'Verify that Feeders at station have a Transformer associated on asset health tool',
                                                         AIStationDF['Suggested_Approach_Station'])



    #Rebuild
    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
     (AIStationDF['Single_Phase_Station'] == True) ,
     'Rebuild 1',
     AIStationDF['Suggested_Approach_Station'])


    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
        (~AIStationDF['FIDequalXFMER']) &
        (AIStationDF['Feeder_Protection'].str.match('Non Sub')) &
        (AIStationDF['Xfmer_Diff_Protection'].str.match('Non Sub')),
        'Rebuild 2',
         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
        (AIStationDF['FIDequalXFMER']) &
        (AIStationDF['Feeder_Protection'].str.match('Non Sub')) &
        (AIStationDF['Xfmer_Diff_Protection'].str.match('Non Sub')) &
        (~AIStationDF['Bus_Equal_XFMER']),
        'Rebuild 3',
        AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Feeder_Protection'] != 'Non Sub') &
                                                         (AIStationDF['Xfmer_Diff_Protection'] != 'Non Sub') &
                                                         (AIStationDF['Mean_Feeder_Age'] > pd.Timedelta(20 * 365)),
                                                         'Rebuild 4',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['Xfmer_Diff_Protection'].str.match('Fused')) &
                                                         (AIStationDF['Feeder_Protection'].isnull())
                                                         ,
                                                         'Rebuild 5',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Xfmer_Diff_Protection'].isin(['Non Sub','Fused'])) &
                                                         (AIStationDF['XFMER_Count'].gt(1)) &
                                                         (~AIStationDF['Bus_Equal_XFMER']),
                                                         'Rebuild 6',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['XFMER_Count'].gt(1)) &
                                                         (~AIStationDF['Bus_Equal_XFMER']),
                                                         'Rebuild 7',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Xfmer_Diff_Protection'] != 'Non Sub') &
                                                         (AIStationDF['XFMER_Count'].gt(1)),
                                                         'Rebuild 8',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['XFMER_Count'].gt(1)),
                                                         'Rebuild 9',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Feeder_Protection'].isnull()),
                                                         'Rebuild 10',
                                                         AIStationDF['Suggested_Approach_Station'])



    #Upgrade
    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Feeder_Protection'] != 'Non Sub') &
                                                         (AIStationDF['Xfmer_Diff_Protection'] != 'Non Sub') &
                                                         (AIStationDF['Mean_Feeder_Age'] < pd.Timedelta(20 * 365)),
                                                         'Component 1',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Bus_Equal_XFMER']),
                                                         'Component 2',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
        (AIStationDF['FIDequalXFMER']) &
        (AIStationDF['Xfmer_Diff_Protection'] != 'Non Sub') &
        (AIStationDF['Feeder_Protection']!= 'Non Sub'),
        'Component 3',
        AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Xfmer_Diff_Protection'] =='Non Sub') &
                                                         (AIStationDF['XFMER_Count'].eq(1)),
                                                         'Component 4',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['Feeder_Protection']) &
                                                         (AIStationDF['XFMER_Count'].eq(1)),
                                                         'Component 5',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['XFMER_Count'].eq(1)),
                                                         'Component 6',
                                                         AIStationDF['Suggested_Approach_Station'])

    AIStationDF['Suggested_Approach_Station'] = np.where((AIStationDF['Suggested_Approach_Station'].str.match('nan')) &
                                                         (~AIStationDF['FIDequalXFMER']) &
                                                         (AIStationDF['XFMER_Count'].eq(1)) &
                                                         (AIStationDF['Xfmer_Diff_Protection'] != 'Non Sub'),
                                                         'Component 7',
                                                         AIStationDF['Suggested_Approach_Station'])




    return AIStationDF

def Match_Missing_Breakers_to_XFMR(Outdoor_BreakerDF, PowerTransformerDF):
    df = PowerTransformerDF[PowerTransformerDF['Station_Name'].isin(PowerTransformerDF['Station_Name'].drop_duplicates(keep=False))]
    df = df.rename(columns={'Maximo_Code':'Associated_XFMR'})
    df = df[df['Station_Name'].isin(Outdoor_BreakerDF[Outdoor_BreakerDF['Associated_XFMR'].isnull()]['Station_Name'])]
    Outdoor_BreakerDF = pd.merge(Outdoor_BreakerDF, df[['Associated_XFMR', 'Station_Name']],on='Station_Name', how='left')
    Outdoor_BreakerDF['Associated_XFMR'] = np.nan
    Outdoor_BreakerDF['Associated_XFMR'] = np.where(Outdoor_BreakerDF['Associated_XFMR_x'].notnull(),
                                                    Outdoor_BreakerDF['Associated_XFMR_x'],
                                                    Outdoor_BreakerDF['Associated_XFMR'])

    Outdoor_BreakerDF['Associated_XFMR'] = np.where(Outdoor_BreakerDF['Associated_XFMR_y'].notnull(),
                                                    Outdoor_BreakerDF['Associated_XFMR_y'],
                                                    Outdoor_BreakerDF['Associated_XFMR'])

    Outdoor_BreakerDF.drop(columns=['Associated_XFMR_x', 'Associated_XFMR_y'])

    return Outdoor_BreakerDF


def main():
    """ Main entry point of the app """
    logger.info("CMPC Wide Area Distribution Main Loop")
    Change_Working_Path('./Data')

    Station_filename = 'Station Location a375a0647.xlsx'
    Transformer_filename = 'Power Transformer Asset a7c07a1cb.xlsx'
    Breaker_filename = 'Breaker Asset a475fe18.xlsx'
    Relay_filename = 'Dist Locations w Relays 110620.xls'
    Circuit_Switcher_filename = 'Circuit Switcher Asset a93a3aebd.xlsx'
    Metalclad_Switchgear_filename = 'Metalclad Switchgear Asset aa554c63f.xlsx'
    Transformer_Risk_filename = 'Oncor Transformer Asset Health Export - Risk Matrix - System.csv'
    Summer_Load_Filename = '2021 Load Projections(4-10)Summer - Clean.xlsx'
    Winter_Load_Filename = '2021 Load Projections(4-10)Winter - Clean.xlsx'
    Fault_Reporting_Proiritization_filename = 'Fault Reporting Prioritization_EDOC.XLSX'
    Fault_Reporting_Proiritization_filename1 = 'WDOC Fault Recording Relay Feeder List with Priority v1.1.xlsx'
    Associated_Breaker_Details_filename = 'Transformer Health - Analysis.xlsx'
    High_Side_Protection = 'Transformer Health - Analysis.xlsx'

    Excel_Files = [Station_filename, Transformer_filename, Breaker_filename, Relay_filename,
                   Metalclad_Switchgear_filename, Summer_Load_Filename, Winter_Load_Filename,
                    Circuit_Switcher_filename]


    #poolo = Pool(processes=15)

    High_Side_Protection_DF = Excel_to_Pandas(Associated_Breaker_Details_filename, check_update=False,
                                                   SheetName='High Side Protection')

    High_Side_Protection_DF = High_Side_Protection_DF[1]

    Associated_Breaker_DetailsDF = Excel_to_Pandas(Associated_Breaker_Details_filename, check_update=False,
                                                   SheetName='Associated Breaker Details')
    Associated_Breaker_DetailsDF = Associated_Breaker_DetailsDF[1]
    # Import Excel files
    #df_listo = pool.map(Excel_to_Pandas, Excel_Files)

    pool = Pool(processes=15)
    df_list = pool.map(smart_excel_to_pandas.smart_excel_to_pandas.Smart_Excel_to_Pandas, Excel_Files)
    pool.close()
    pool.join()

    Excel_Files = [Fault_Reporting_Proiritization_filename, Fault_Reporting_Proiritization_filename1]

    pool = Pool(processes=15)
    #df_list.append(pool.starmap(smart_excel_to_pandas.smart_excel_to_pandas.Smart_Excel_to_Pandas,
     #                           [(Excel_Files[0],None), (Excel_Files[1],None)]))
    df_list1 = pool.map(Excel_to_Pandas, Excel_Files)

    df_list = df_list + df_list1
    pool.close()
    pool.join()

    #df_list[next(i for i, t in enumerate(df_list) if t[0] == Fault_Reporting_Proiritization_filename)][1] = pd.concat([
    df = pd.concat([
       df_list[next(i for i, t in enumerate(df_list) if t[0] == Fault_Reporting_Proiritization_filename)][1],
       df_list[next(i for i, t in enumerate(df_list) if t[0] == Fault_Reporting_Proiritization_filename1)][1]], axis=0, ignore_index=True)
    tup = (Fault_Reporting_Proiritization_filename, df)
    df_list = df_list + [tup]
    del df_list[10]


    Transformer_RiskDF = Cleanup_Dataframe(pd.read_csv(Transformer_Risk_filename))

    # Data Cleanup

    AIStationDF = station_df_cleanup(df_list[next(i for i, t in enumerate(df_list) if t[0] == Station_filename)][1],
                                     df_list[next(
                                         i for i, t in enumerate(df_list) if t[0] == Metalclad_Switchgear_filename)][1])

    PowerTransformerDF = transformer_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Transformer_filename)][1])
    Outdoor_BreakerDF = breaker_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Breaker_filename)][1])
    RelayDataDF = relay_df_cleanup(df_list[next(i for i, t in enumerate(df_list) if t[0] == Relay_filename)][1])
    Summer_LoadDF = summer_load_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Summer_Load_Filename)][1])

    Winter_LoadDF = summer_load_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Winter_Load_Filename)][1])

    Fault_Reporting_ProiritizationDF = FRP.Fault_Reporting_Proiritization_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Fault_Reporting_Proiritization_filename)][1])

    Circuit_Switcher_df = CS_df_cleanup(
        df_list[next(i for i, t in enumerate(df_list) if t[0] == Circuit_Switcher_filename)][1])

    # Create new date in the dataframes
    Fault_Reporting_ProiritizationDF = FRP.Fault_Reporting_Proiritization_df_create_data(
        Fault_Reporting_ProiritizationDF)
    Summer_LoadDF = summer_load_df_create_data(Summer_LoadDF, AIStationDF)
    Winter_LoadDF = summer_load_df_create_data(Winter_LoadDF, AIStationDF)
    AIStationDF = station_df_create_data(AIStationDF, PowerTransformerDF, Outdoor_BreakerDF)
    PowerTransformerDF = transformer_df_create_data(PowerTransformerDF, Transformer_RiskDF, Summer_LoadDF,
                                                    Winter_LoadDF, AIStationDF)
    Outdoor_BreakerDF = breaker_df_create_data(Outdoor_BreakerDF, PowerTransformerDF, Fault_Reporting_ProiritizationDF)
    Outdoor_BreakerDF = Add_Associated_XMR_Details(Outdoor_BreakerDF, Associated_Breaker_DetailsDF)
    Outdoor_BreakerDF = Match_Missing_Breakers_to_XFMR(Outdoor_BreakerDF, PowerTransformerDF)
    RelayDataDF = relay_df_create_data(RelayDataDF, PowerTransformerDF, Outdoor_BreakerDF)
    AIStationDF = add_Risk_to_Stationdf(AIStationDF, PowerTransformerDF)
    AIStationDF = add_MVA_Exceeded_Stationdf(AIStationDF, PowerTransformerDF)

    #Outdoor_BreakerDF = Outdoor_Breaker.add_Relay_Outdoor_BreakerDF(RelayDataDF, Outdoor_BreakerDF)
    Outdoor_BreakerDF = Outdoor_Breaker.add_Relay2_Outdoor_BreakerDF(RelayDataDF, Outdoor_BreakerDF)
    PowerTransformerDF = Add_fused_Bank_to_PowerTransformerDF(PowerTransformerDF, RelayDataDF)



    AIStationDF = Add_Fused_Bank_to_Stationdf(AIStationDF, PowerTransformerDF)
    PowerTransformerDF = PowerTransformer.Add_Feeder_Protection_on_Bank(PowerTransformerDF, Outdoor_BreakerDF)
    AIStationDF = Add_Feeder_Protection_on_Station(PowerTransformerDF, AIStationDF)
    AIStationDF = Add_Bus_Tie_at_Station(AIStationDF, Outdoor_BreakerDF)
    #PowerTransformerDF = PowerTransformer.add_Xfmer_Diff_Protection_PowerTransformerDF(RelayDataDF, PowerTransformerDF)
    PowerTransformerDF = PowerTransformer.add_Xfmer2_Diff_Protection_PowerTransformerDF(RelayDataDF, PowerTransformerDF)
    PowerTransformerDF = PowerTransformer.Add_High_Side_Interrupter_PowerTransformerDF(PowerTransformerDF,
                                                                                       High_Side_Protection_DF)
    AIStationDF = Add_Xfmer_Diff_Protection_on_Station(PowerTransformerDF, AIStationDF)
    AIStationDF = Add_CS_count_on_Station(Circuit_Switcher_df, AIStationDF)
    AIStationDF = Add_FID_count_equal_XFMER_count(AIStationDF)
    AIStationDF = Suggested_Approach_Station(AIStationDF)
    PowerTransformerDF = PowerTransformer.Suggested_Approach_Bank(PowerTransformerDF)

    # Analytics
    df = PowerTransformerDF.groupby(['Feeder_Protection', 'Xfmer_Diff_Protection', 'High_Side_Interrupter', 'Suggested_Approach_Bank'],
                             dropna=False).size().reset_index().rename(columns={0: 'count'})
    df.to_excel('Bank Analytics.xlsx')


    df = AIStationDF.groupby(['Single_Phase_Station', 'FIDequalXFMER', 'Xfmer_Diff_Protection', 'Bus_Equal_XFMER',
                              'Feeder_Protection', 'Suggested_Approach_Station'], dropna=False).size().reset_index().rename(columns={0:'count'})
    df.to_excel('Analytics.xlsx')
    # Select columns to keep
    # AIStationDF = AIStationDF[
    #   ['Region', 'Work_Center', 'Maximo_Code', 'Station_Name', 'STATION_STR_TYPE', 'Age', 'Single_Phase_Station',
    #   'Has_Fused_Bank', 'XFMER_Count', 'BUS_TIE_Count',  'Max_Risk_Index_at_Station', 'Max_MVA_Exceeded', 'Mean_Feeder_Age', 'Feeder_Protection'
    #  ]]

    AIStationDF = AIStationDF[['Region', 'Work_Center', 'Maximo_Code', 'Station_Name', 'STATION_STR_TYPE', 'Age',
                               'Single_Phase_Station', 'Has_Fused_Bank', 'XFMER_Count', 'FID_Count', 'FIDequalXFMER', 'BUS_TIE_Count', 'Bus_Equal_XFMER', 'Mean_Feeder_Age', 'Feeder_Protection',
         'Xfmer_Diff_Protection', 'Suggested_Approach_Station'
         ]]

    PowerTransformerDF = PowerTransformerDF[['Region', 'Work_Center', 'Station_Name', 'Maximo_Code',
                                             'Age', 'MAXIMUM_MVA', 'LV_NOM_KV', 'Risk_Index_(Normalized)',
                                             'Max_Projected_Summer_Load', 'Max_Projected_Winter_Load',
                                             'Max_MVA_Exceeded', 'NUM_PH', 'IsFused', 'Feeder_Protection',
                                             'Xfmer_Diff_Protection', 'High_Side_Interrupter'
                                             ]]

    Outdoor_BreakerDF = Outdoor_BreakerDF[['Region', 'Work_Center', 'Station_Name', 'Maximo_Code', 'Age',
                                           'BKR_SERVICE', 'SELF_CONTAINED', 'Manufacturer', 'BKR_MECH_MOD',
                                           'BKR_INTERR', 'Associated_XFMR', 'DOC_Fault_Reporting_Prioritization',
                                           'Feeder_Protection']]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('CMPC_WideArea_AIS.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    AIStationDF.to_excel(writer, sheet_name='Stations', index=False)
    PowerTransformerDF.to_excel(writer, sheet_name='Transformers', index=False)
    Outdoor_BreakerDF.to_excel(writer, sheet_name='Outdoor Breakers', index=False)
    RelayDataDF.to_excel(writer, sheet_name='Relay', index=False)
    Summer_LoadDF.to_excel(writer, sheet_name='Summer Load', index=False)
    Winter_LoadDF.to_excel(writer, sheet_name='Winter Load', index=False)
    Circuit_Switcher_df.to_excel(writer, sheet_name='Circuit Switcher', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """ This is executed when run from the command line """
    # Setup Logging
    logger = logging.getLogger('root')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)

    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
