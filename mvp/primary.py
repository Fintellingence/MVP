import pandas as pd
import mvp

class PrimaryModel():

    def __init__(self, raw_data, model_type, parameters):
        self.model_type = model_type
        for key,value in parameters.items():
            self.key = value
        self.feature_data = mvp.curated.CuratedData(raw_data, parameters['ModelParameters'])

