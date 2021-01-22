import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import logging
logger = logging.getLogger('root')


def Add_colunms_to_df(df, list_of_column_names):
    Physical_Design_Factors_list = ['Construction_RAW', 'Bus_RAW', 'Insulators_RAW',  'Property_RAW']
    for name in Physical_Design_Factors_list:
        df[name] = np.nan
    return df

def station_df_cleanup(StationDF, Metalclad_Switchgear_DF):
    # Filters
    StationDF = StationDF[StationDF['Ownership_Name'].str.contains('ONCOR ELECTRIC DELIVERY')]
    StationDF = StationDF[StationDF['STATION_CLASS'].str.contains('SUBSTATION')]
    StationDF = StationDF[~StationDF['Station_Name'].isin(Metalclad_Switchgear_DF['Station_Name'])]

    # Convert Datatypes
    StationDF['IN_SERVICE_DATE'] = pd.to_datetime(StationDF['IN_SERVICE_DATE'], errors='raise')

    return StationDF

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

def Suggested_Approach_Station2(stationdf, XFMR_df):
    stationdf['Suggested_Approach_Station2'] = np.nan
    XFMR_df['Suggested_Approach_Bank'] = np.where(XFMR_df['Suggested_Approach_Bank'].str.contains('Rebuild'),
                                                   'Rebuild',
                                                   XFMR_df['Suggested_Approach_Bank'])

    XFMR_df['Suggested_Approach_Bank'] = np.where(XFMR_df['Suggested_Approach_Bank'].str.contains('Upgrade'),
                                                   'Upgrade',
                                                   XFMR_df['Suggested_Approach_Bank'])

    XFMR_df['Suggested_Approach_Bank'] = np.where(
        XFMR_df['Suggested_Approach_Bank'].str.contains('Data Validation Needed'),
        'Data Validation Needed',
        XFMR_df['Suggested_Approach_Bank'])

    XFMR_df['Age'] = XFMR_df['Age'].fillna(pd.Timedelta(seconds=0))

    station_array = stationdf['Station_Name'].unique()
    for station in tqdm(station_array):
        filtered = XFMR_df[XFMR_df['Station_Name'] == station]
        if len(filtered) >= 1:
            logger.info(str(station) + ' has transformer')
            if filtered['Suggested_Approach_Bank'].nunique() == 1:
                approach = filtered['Suggested_Approach_Bank'].unique()[0]
                stationdf['Suggested_Approach_Station2'] = np.where(stationdf['Station_Name'] == station, approach, stationdf['Suggested_Approach_Station2'])
            elif (filtered['Suggested_Approach_Bank'].nunique() >> 1 and bool(filtered['Age'].mean().days >> 40)):
                logger.info(str(station) + ' is transformer is old and will be rebuilt')
                stationdf['Suggested_Approach_Station2'] = np.where(stationdf['Station_Name'] == station,
                                                                    'Rebuild',
                                                                    stationdf['Suggested_Approach_Station2'])
            elif (filtered['Suggested_Approach_Bank'].nunique() >> 1 and bool(filtered['Age'].mean().days << 40)):
                logger.info(str(station) + ' is transformer is not old enough so will be upgraded')
                stationdf['Suggested_Approach_Station2'] = np.where(stationdf['Station_Name'] == station,
                                                                    'Upgrade',
                                                                    stationdf['Suggested_Approach_Station2'])
        else:
            logger.info(str(station) + ' has no transformer')
            stationdf['Suggested_Approach_Station2'] = np.where(stationdf['Station_Name'] == station,
                                                                'Verify that station has no Transformer',
                                                                stationdf['Suggested_Approach_Station2'])


    return stationdf

def Add_Needed_MVA_Station(AIStationDF, PowerTransformerDF):

    AIStationDF['Needed_MVA'] = 0
    Station_MVA = PowerTransformerDF.groupby(['Station_Name'])['MAXIMUM_MVA'].sum()
    Station_Projected_Summer_Load = PowerTransformerDF.groupby(['Station_Name'])['Max_Projected_Summer_Load'].sum()
    StationProjected_Winter_Load = PowerTransformerDF.groupby(['Station_Name'])['Max_Projected_Winter_Load'].sum()
    AIStationDF = pd.merge(AIStationDF, Station_MVA, on='Station_Name', how='left')
    AIStationDF = pd.merge(AIStationDF, Station_Projected_Summer_Load, on='Station_Name', how='left')
    AIStationDF = pd.merge(AIStationDF, StationProjected_Winter_Load, on='Station_Name', how='left')

    AIStationDF['Needed_MVA'] = np.where(AIStationDF['Max_Projected_Summer_Load'].isnull() &
                              AIStationDF['Max_Projected_Winter_Load'].isnull(),
                              AIStationDF['MAXIMUM_MVA'],
                              AIStationDF['Needed_MVA']
                                       )

    AIStationDF['Needed_MVA'] = np.where(AIStationDF['Max_Projected_Summer_Load'].notna() &
                                       AIStationDF['Max_Projected_Winter_Load'].notna() &
                                       AIStationDF['Max_Projected_Summer_Load'].ge(AIStationDF['Max_Projected_Winter_Load']),
                                       AIStationDF['Max_Projected_Summer_Load'],
                                       AIStationDF['Needed_MVA']
                                       )

    AIStationDF['Needed_MVA'] = np.where(AIStationDF['Max_Projected_Summer_Load'].notna() &
                                       AIStationDF['Max_Projected_Winter_Load'].notna() &
                                       AIStationDF['Max_Projected_Summer_Load'].le(AIStationDF[
                                           'Max_Projected_Winter_Load']),
                                       AIStationDF['Max_Projected_Winter_Load'],
                                       AIStationDF['Needed_MVA']
                                       )


    return AIStationDF

def Add_Suggested_Solution_Station(AIStationDF):
    AIStationDF['Suggested_Solution'] = ''

    AIStationDF['Suggested_Solution'] = np.where(AIStationDF['Suggested_Approach_Station2'].str.match('Rebuild') &
                                                 AIStationDF['Needed_MVA'].lt(7.5),
                                                 'Rural',
                                                 AIStationDF['Suggested_Solution'])

    AIStationDF['Suggested_Solution'] = np.where(AIStationDF['Suggested_Approach_Station2'].str.match('Rebuild') &
                                                 AIStationDF['Needed_MVA'].ge(7.5) &
                                                 AIStationDF['Needed_MVA'].lt(28),
                                                 'SUB IV (Little)',
                                                 AIStationDF['Suggested_Solution'])

    AIStationDF['Suggested_Solution'] = np.where(AIStationDF['Suggested_Approach_Station2'].str.match('Rebuild') &
                                                 AIStationDF['Needed_MVA'].ge(28),
                                                  'SUB IV (Big)',
                                                 AIStationDF['Suggested_Solution'])

    return AIStationDF




