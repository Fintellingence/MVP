import os
from functools import partial

import numpy as np
import pandas as pd

from mvp.selection import KFold, count_occurrences, avg_uniqueness

class MetaModel:
    def __init__(self, model, n_split_cv=5, n_split_gs=5, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}


    def bagging_model(self):
        pass

    def random_forest(self):
        pass
