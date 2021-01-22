def Add_colunms_to_df(df, list_of_column_names):
    Physical_Design_Factors_list = ['Construction_RAW', 'Bus_RAW', 'Insulators_RAW',  'Property_RAW']
    for name in Physical_Design_Factors_list:
        df[name] = np.nan
    return df