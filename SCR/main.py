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
from multiprocessing import Pool
import FRP


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
    return relaydf


def relay_df_create_data(relaydf):
    relaydf['Maximo_Asset_Protected_Station'] = relaydf['LOCATION'].str.slice(start=0, stop=5)
    relaydf['Maximo_Asset_Protected_Device_Type'] = relaydf['LOCATION'].str.slice(start=10, stop=3)
    relaydf['Maximo_Asset_Protected_Device_Num'] = relaydf['LOCATION'].str.slice(start=13)

    relaydf['Maximo_Asset_Protected'] = np.where(relaydf['Maximo_Asset_Protected_Device_Type'] == 'BKR', relaydf[
    'Maximo_Asset_Protected_Station'] + '-0', relaydf['Maximo_Asset_Protected_Device_Num'])
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


def add_Relay_Stationdf(AIStationDF, RelayDataDF, Outdoor_BreakerDF):
    RelayDataDF_351_feeders = RelayDataDF[(RelayDataDF['MODEL'].str.contains('351')) &
                                          RelayDataDF['MFG'].str.match('SEL')]
    RelayDataDF_351_feeders['SUB_4_Protection'] = True
    Outdoor_BreakerDF = pd.merge(Outdoor_BreakerDF,
                                 RelayDataDF_351_feeders[['Maximo_Asset_Protected', 'SUB_4_Protection']],
                                 left_on='Maximo_Code', right_on='Maximo_Asset_Protected', how='left')

    return Outdoor_BreakerDF


def Add_Associated_XMR_Details(Outdoor_BreakerDF, Associated_Breaker_DetailsDF):
    Associated_Breaker_DetailsDF = Associated_Breaker_DetailsDF.rename(
        columns={"Maximo_Code": "Associated_XFMR", "Associated_Breaker_Maximo_Code": "Maximo_Code"})
    Outdoor_BreakerDF = pd.merge(Outdoor_BreakerDF, Associated_Breaker_DetailsDF[['Maximo_Code', 'Associated_XFMR']],
                                 on='Maximo_Code', how='left')

    return Outdoor_BreakerDF


def main():
    """ Main entry point of the app """
    logger.info("CMPC Wide Area Distrution Main Loop")
    Change_Working_Path('../Data')

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

    Excel_Files = [Station_filename, Transformer_filename, Breaker_filename, Relay_filename,
                   Metalclad_Switchgear_filename, Summer_Load_Filename, Winter_Load_Filename,
                   Fault_Reporting_Proiritization_filename, Fault_Reporting_Proiritization_filename1]

    pool = Pool(processes=15)
    Associated_Breaker_DetailsDF = Excel_to_Pandas(Associated_Breaker_Details_filename, check_update=False,
                                                   SheetName='Associated Breaker Details')
    Associated_Breaker_DetailsDF = Associated_Breaker_DetailsDF[1]
    # Import Excel files
    df_list = pool.map(Excel_to_Pandas, Excel_Files)

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
    RelayDataDF = relay_df_create_data(RelayDataDF)
    AIStationDF = add_Risk_to_Stationdf(AIStationDF, PowerTransformerDF)
    AIStationDF = add_MVA_Exceeded_Stationdf(AIStationDF, PowerTransformerDF)
    Outdoor_BreakerDF = add_Relay_Stationdf(AIStationDF, RelayDataDF, Outdoor_BreakerDF)

    # Select columns to keep
    AIStationDF = AIStationDF[
        ['Region', 'Work_Center', 'Maximo_Code', 'Station_Name', 'STATION_STR_TYPE', 'Age', 'Single_Phase_Station',
         'XFMER_Count', 'Max_Risk_Index_at_Station', 'Max_MVA_Exceeded', 'Mean_Feeder_Age'
         ]]

    PowerTransformerDF = PowerTransformerDF[['Region', 'Work_Center', 'Station_Name', 'Maximo_Code',
                                             'Age', 'MAXIMUM_MVA', 'LV_NOM_KV', 'Risk_Index_(Normalized)',
                                             'Max_Projected_Summer_Load', 'Max_Projected_Winter_Load',
                                             'Max_MVA_Exceeded']]

    Outdoor_BreakerDF = Outdoor_BreakerDF[['Region', 'Work_Center', 'Station_Name', 'Maximo_Code', 'Age',
                                           'BKR_SERVICE', 'SELF_CONTAINED', 'Manufacturer', 'BKR_MECH_MOD',
                                           'BKR_INTERR', 'Associated_XFMR', 'DOC_Fault_Reporting_Prioritization',
                                           'SUB_4_Protection']]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('../CMPC_WideArea_AIS.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    AIStationDF.to_excel(writer, sheet_name='Stations', index=False)
    PowerTransformerDF.to_excel(writer, sheet_name='Transformers', index=False)
    Outdoor_BreakerDF.to_excel(writer, sheet_name='Outdoor Breakers', index=False)
    RelayDataDF.to_excel(writer, sheet_name='Relay', index=False)
    Summer_LoadDF.to_excel(writer, sheet_name='Summer Load', index=False)
    Winter_LoadDF.to_excel(writer, sheet_name='Winter Load', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """ This is executed when run from the command line """
    # Setup Logging
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)

    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
