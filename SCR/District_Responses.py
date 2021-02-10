import pandas as pd
import xlsxwriter
import logging


def write_row(sheet, list, row=0):
    for index, value in enumerate(list):
        sheet.write(row, index, value)
    return True


def write_dropdown(ws, column, row, list, default='Select a value from a drop down list'):
    txt = default

    ws.write(column, row, txt)
    ws.data_validation(column, row, column, row, {'validate': 'list',
                                                  'source': list})
    return True

def write_relay_cell(wb, ws, row, column, vaule, Dev_Status):
    cell_format = wb.add_format()
    cell_format.set_pattern(1)
    if Dev_Status == 'Operating':
        cell_format.set_bg_color('green')
    elif Dev_Status == 'Operating Needs Review':
        cell_format.set_bg_color('red')
    elif Dev_Status == 'Planned':
        cell_format.set_bg_color('yellow')

    ws.write(row, column, vaule, cell_format)

def write_data_to_sheet(wb, ws, df):
    Con_Options = ['Wood', 'Lattice']
    Bus_Options = ['Angle Iron', 'Wire Conductor', 'Copper']
    Insulator_Options = ['Ball and Pin', 'Bell']
    Mobile_Options = ['Adequate', 'Inadequate']
    Site_Options = ['Issue', 'Non Issue']

    Con_Options.sort()
    Bus_Options.sort()
    Insulator_Options.sort()
    Mobile_Options.sort()
    Site_Options.sort()

    for index, station in enumerate(df.Station_Name.values):
        row_data = [df.Station_Name.iloc[index], df.Maximo_Code.iloc[index], '', '',
                    '',
                    '',
                    '',
                    df.High_Side_Interrupter.iloc[index],
                    '',
                    ''
                    ]
        write_row(ws, row_data, 1 + index)

        write_relay_cell(wb, ws, 1 + index, 8, df.Xfmer_Diff_Protection.iloc[index], df.XFMER_DEV_STATUS.iloc[index])
        write_relay_cell(wb, ws, 1 + index, 9, df.Feeder_Protection.iloc[index], df.XFMER_DEV_STATUS.iloc[index])

        write_dropdown(ws, 1 + index, 2, Con_Options)
        write_dropdown(ws, 1 + index, 3, Bus_Options)
        write_dropdown(ws, 1 + index, 4, Insulator_Options)
        write_dropdown(ws, 1 + index, 5, Mobile_Options)
        write_dropdown(ws, 1 + index, 6, Site_Options)
    return True


def main(df):

    for work_center in  df.Work_Center.unique():
        workcenter_df = df.query('Work_Center == @work_center')

        workbook = xlsxwriter.Workbook('../Data/' + work_center + '.xlsx')
        worksheet = workbook.add_worksheet()

        headers = ['Station Name', 'Maximo Code', 'Station Construction', 'Bus', 'Insulators', 'Mobile Equipment Space',
                   'Site Grading', 'Highside Protection', 'XFMER Relaying', 'Feeder Relaying', 'Comments']
        write_row(worksheet, headers)

        write_data_to_sheet(workbook, worksheet, workcenter_df)

        workbook.close()


if __name__ == '__main__':
    """ This is executed when run from the command line """
    # Setup Logging
    logger = logging.getLogger('root')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)

    district = ['Graham', 'Graham', 'Graham']
    Stations = ['sta 1', 'sta 2', 'sta 2']
    Banks = ['BNK1', 'BNK2', 'BNK3']
    HS = ['Breaker', 'Feeder', 'Switch']
    XFMER_Relaying = ['non-mod', 'Broken', 'Fixed']
    FD_Relaying = ['Broken', 'non-mod', 'Fixed2']
    XFMER_DEV_STATUS = ['Operating', 'Operating Needs Review', 'Planned']
    FD_DEV_STATUS = ['Planned', 'Operating', 'Operating Needs Review']

    data_tuple = list(zip(district, Stations, Banks, HS, XFMER_Relaying, FD_Relaying, XFMER_DEV_STATUS, FD_DEV_STATUS ))

    df = pd.DataFrame(data_tuple,
                   columns=['Work_Center', 'Station_Name', 'Maximo_Code', 'High_Side_Interrupter', 'Xfmer_Diff_Protection',
                            'Feeder_Protection', 'XFMER_DEV_STATUS', 'FD_DEV_STATUS'])

    main(df)
