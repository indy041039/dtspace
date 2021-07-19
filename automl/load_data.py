import pandas as pd
import numpy as np
import random

def load_csv(csv_path, dev=False, sample_ratio=None, **kwargs):
    if not dev:
        df = pd.read_csv(csv_path, **kwargs)
    else:
        df = pd.read_csv(csv_path,
                         skiprows=lambda i: i>0 and random.random() > sample_ratio,
                         **kwargs)
    return df

# def load_sql(sql_path, dev=False, sample_ratio=None, **kwargs):
#     if not dev:
#         df = pd.read_csv(csv_path, **kwargs)
#     else:
#         df = pd.read_csv(csv_path,
#                          skiprows=lambda i: i>0 and random.random() > sample_ratio,
#                          **kwargs)
#     return df