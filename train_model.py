#-*- coding:utf-8 -*-
"""
This file implements the training and testing of the model, 

requiring changes to the data and model Settings in the previous configuration.

"""
import numpy as np
import pandas as pd
import multiprocessing
import random
import os
import config
import logging
# Set the format and level of log.
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
from frameworks.utils import *
from sklearn.externals import joblib


if __name__ == "__main__":
    settings = ExperimentSettings(al_name="PUAD", init_label_count=8, new_kpi_train_ratio=0.4, if_save_label_index=True)
    # exp = SupervisedExperiment(settings)
    exp = PUADExperiment(settings)
    