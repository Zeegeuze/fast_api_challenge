import numpy as np
import time
import pickle

from colorama import Fore, Style
from typing import Tuple

# def initialize_model(input_shape: tuple) -> Model:
#     """
#     Initialize the Neural Network with random weights
#     """
#     reg = regularizers.l1_l2(l2=0.005)

#     model = Sequential()

#     print("✅ Model initialized")

#     return model

def load_model():
    return pickle.load(open("immo_eliza/ml_logic/model.pkl", "rb"))
