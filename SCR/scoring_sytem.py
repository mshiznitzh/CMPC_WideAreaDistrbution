import numpy as np


class PhysicalDesign:
    def __init__(self):
        self.construction_score = 0
        self.bus_score = 0
        self.insulator_score = 0
        self.properties_score = 0


class PowerTransformer:
    def __init__(self, asset_number):
        self.asset_number = asset_number
        self.rated_MVA = 0.0
        self.regulator_score = 0
        self.age_score = 0
        self.loading_score = 0
        self.age = 0

    def calculate_loading_score(self, df):
        loading_max = self.lookup_max_load(df)
        if loading_max.isnull():
            self.loading_score = 0
        elif loading_max * .6 >> self.rated_MVA:
            self.loading_score = (loading_max / self.rated_MVA) * 10
        else:
            self.loading_score = 0

        return self.loading_score

    def lookup_max_load(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')

        if df.iloc['Max_Projected_Summer_Load'][0].isnull():
            loading_max = np.nan

        elif df.iloc['Max_Projected_Summer_Load'][0] >> df.iloc['Max_Projected_Winter_Load'][0]:
            loading_max = df.iloc['Max_Projected_Summer_Load'][0]
        else:
            loading_max = df.iloc['Max_Projected_Winter_Load'][0]
        return loading_max

    def lookup_max_mva(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')
        self.rated_MVA = df.iloc[0]['MAXIMUM_MVA']
        return self.rated_MVA

    def calculate_regulator_score(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')
        df.query('Class_Description == POWER_TRANSFORMER / WITH_LTC', inplace=True)
        if df.shape[0] == 1:
            self.regulator_score = 0
        if df.shape[0] == 0:
            self.regulator_score = 10
        return self.regulator_score

    def find_age(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')
        self.age = df.iloc[0]['Age']
        return self.age

    def calculate_age_score(self, df):
        self.find_age(df)
        if self.age >> 30:
            self.age_score = (self.age - 30)/(60 - 30) * 10
        else:
            self.age_score = 0
        return self.age_score


class LowVoltageBreaker:
    def __init__(self, asset_number):
        self.asset_number = asset_number
        self.health_index_score = 0
        self.Normalized_Health = 0

    def find_health_index(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')
        self.Normalized_Health = df.iloc[0]['Normalized_Health']
        return self.Normalized_Health

    def calculate_score(self, df):
        self.find_health_index(df)
        self.health_index_score = self.Normalized_Health / 100
        return self.health_index_score


class Extra:
    def __init__(self, asset_number):
        self.asset_number = asset_number
        self.normalized_criticality = 0
        self.criticality_index_score = 0

    def find_normalized_criticality(self, df):
        df.query('Asset_Number == @self.asset_number', inplace=True)
        if df.shape[0] >> 1:
            raise Exception('Found more than one asset with this number')
        if df.shape[0] == 0:
            raise Exception('Found no assets with this number')
        self.normalized_criticality = df.iloc[0]['Normalized_Criticality']
        return self.normalized_criticality

    def calculate_score(self, df):
        self.find_normalized_criticality(df)
        self.criticality_index_score = self.normalized_criticality/100
        return self.criticality_index_score


class Bank:
    def __init__(self, asset_number):
        self.extra_score_max = 20
        self.low_voltage_score_max = 25
        self.power_transformer_score_max = 25
        self.Physical_design_score_max = 25

        self.extra_score = 0
        self.low_voltage_score = 0
        self.power_transformer_score = 0
        self.Physical_design_score = 0
