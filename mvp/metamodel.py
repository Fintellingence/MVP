import os
import logging
import logging.handlers
from functools import partial

import numpy as np
import pandas as pd


class MetaModel:
    def __init__(self):
        os.makedirs("logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            "logs/metamodel.log", maxBytes=200 * 1024 * 1024, backupCount=1
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)

    def bagging_model(self):
        pass

    def random_forest(self):
        pass
