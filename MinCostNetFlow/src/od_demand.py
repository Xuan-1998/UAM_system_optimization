import numpy as np
import pandas as pd

def parse_demand(path):
    demand = pd.read_csv(path)
    