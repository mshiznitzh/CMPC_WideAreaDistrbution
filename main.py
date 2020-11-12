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
    # Check if New path exists
    if os.path.exists(path):
        # Change the current working Directory
        try:
            os.chdir(path)  # Change the working directory
        except OSError:
            logger.error("Can't change the Current Working Directory", exc_info = True)
    else:
        print("Can't change the Current Working Directory because this path doesn't exits")

#Pandas Functions
def Excel_to_Pandas(filename,check_update=False,SheetName=None):
    logger.info('importing file ' + filename)
    df=[]
    if check_update == True:
        timestamp = dt.datetime.fromtimestamp(Path(filename).stat().st_mtime)
        if dt.datetime.today().date() != timestamp.date():
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(title =' '.join(['Select file for', filename]))

    try:
        df = pd.read_excel(filename, SheetName)
        df = pd.concat(df, axis=0, ignore_index=True)
    except:
        logger.error("Error importing file " + filename, exc_info=True)

    df=Cleanup_Dataframe(df)
    logger.info(df.info(verbose=True))
    return df

def Cleanup_Dataframe(df):
    logger.info('Starting Cleanup_Dataframe')
    logger.debug(df.info(verbose=False))
    # Remove whitespace on both ends of column headers
    df.columns = df.columns.str.strip()

    # Replace whitespace in column header with _
    df.columns = df.columns.str.replace(' ', '_')

    return df

def station_df_cleanup(StationDF, Metalclad_Switchgear_DF):
#Filters
    StationDF = StationDF[StationDF['Ownership_Name'].str.contains('ONCOR ELECTRIC DELIVERY')]
    StationDF = StationDF[StationDF['STATION_CLASS'].str.contains('SUBSTATION')]
    StationDF = StationDF[~StationDF['Station_Name'].isin(Metalclad_Switchgear_DF['Station_Name'])]

#Convert Datatypes
    StationDF['IN_SERVICE_DATE'] = pd.to_datetime(StationDF['IN_SERVICE_DATE'], errors='raise')

    return StationDF

def transformer_df_cleanup(TransformerDF):
    TransformerDF = TransformerDF[TransformerDF['XFMR_SERVICE'].str.contains('POWER')]

    return TransformerDF

def station_df_create_data(StationDF, PowerTransformerDF, Outdoor_BreakerDF):
    StationDF['Age'] = (dt.date.today() - StationDF['IN_SERVICE_DATE'].dt.date)/365

#Get a count of the number of XMFR at a station
    countDF = PowerTransformerDF.groupby(['Station_Name'], as_index=False).agg(XFMER_Count=('Asset_Number', pd.Series.count))
#Merge count into StationDF
    StationDF = pd.merge(StationDF, countDF, on='Station_Name', how='left', )

    return StationDF

def breaker_df_cleanup(BreakerDF):
    BreakerDF = BreakerDF[BreakerDF['BKR_CLASS'].str.contains('OUTDOOR')]

    return BreakerDF

def main():

    """ Main entry point of the app """
    logger.info("CMPC Wide Area Distrution Main Loop")
    Change_Working_Path('./Data')

    Station_filename = 'Station Location a375a0647.xlsx'
    Transformer_filename = 'Power Transformer Asset a7c07a1cb.xlsx'
    Breaker_filename = 'Breaker Asset a475fe18.xlsx'
    Relay_filename = 'Dist Locations w Relays 110620.xls'
    Circuit_Switcher_filename = 'Circuit Switcher Asset a93a3aebd.xlsx'
    Metalclad_Switchgear_filename = 'Metalclad Switchgear Asset aa554c63f.xlsx'

    #Import Excel files
    StationDF = Excel_to_Pandas(Station_filename, False)
    TransformerDF = Excel_to_Pandas(Transformer_filename, False)
    BreakerDF = Excel_to_Pandas(Breaker_filename, False)
    RelayDF = Excel_to_Pandas(Relay_filename, False)
    Metalclad_Switchgear_DF = Excel_to_Pandas(Metalclad_Switchgear_filename, False)

    #Data Cleanup
    AIStationDF = station_df_cleanup(StationDF, Metalclad_Switchgear_DF)
    PowerTransformerDF = transformer_df_cleanup(TransformerDF)
    Outdoor_BreakerDF = breaker_df_cleanup(BreakerDF)

    #Create new date in the dataframes
    AIStationDF = station_df_create_data(AIStationDF, PowerTransformerDF, Outdoor_BreakerDF)

    # Select columns to keep
    AIStationDF = AIStationDF[['Region', 'Work_Center', 'Maximo_Code', 'Station_Name', 'STATION_STR_TYPE', 'Age','XFMER_Count'
                           ]]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('CMPC_WideArea_AIS.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    AIStationDF.to_excel(writer, sheet_name='Stations', index=False)
    PowerTransformerDF.to_excel(writer, sheet_name='Transformers', index=False)
    Outdoor_BreakerDF.to_excel(writer, sheet_name='Outdoor Breakers', index=False)

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
