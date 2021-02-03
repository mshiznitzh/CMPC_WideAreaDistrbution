import SCR.scoring_sytem as ss
import pandas as pd


class TestExtra:
    def test_find_normalized_criticality_one(self):
        data = [[255347, 77.78], [255402, 88.89], [254008, 70.37]]
        df = pd.DataFrame(data, columns=['Asset_Number', 'Normalized_Criticality'])
        obj = ss.Extra(255347)
        assert obj.find_normalized_criticality(df) == 77.78

    def test_calculate_score_one(self):
        data = [[255347, 77.78], [255402, 88.89], [254008, 70.37]]
        df = pd.DataFrame(data, columns=['Asset_Number', 'Normalized_Criticality'])
        obj = ss.Extra(255347)
        assert obj.calculate_score(df) == 77.78/100


class TestLowVoltageBreaker:
    def test_find_health_index_one(self):
        data = [[135634, 51.22], [134092, 58.54], [133164, 48.78]]
        df = pd.DataFrame(data, columns=['Asset_Number', 'Normalized_Health'])
        obj = ss.LowVoltageBreaker(135634)

        assert obj.find_health_index(df) == 51.22

    def test_calculate_score_one(self):
        data = [[135634, 51.22], [134092, 58.54], [133164, 48.78]]
        df = pd.DataFrame(data, columns=['Asset_Number', 'Normalized_Health'])
        obj = ss.LowVoltageBreaker(134092)
        assert obj.calculate_score(df) == 58.54/100
