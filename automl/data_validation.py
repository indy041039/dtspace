from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

import re

from sklearn.base import TransformerMixin, BaseEstimator
from collections import OrderedDict

class ValidateSchema():
  '''
  Transform inference data into the same schema and order as training data
  '''
  def __init__(self, mode='error'):
    self.schema = {}
    self.mode = mode

  def _check_schema(self, X):
    # schema = X.dtypes
    # schema = {k:re.sub('[0-9]','', v.name) for k,v in schema.items()}
    return X.dtypes

  def infer_schema(self, X, y=None):
    X = X.copy()
    self.schema = self._check_schema(X)
    self.order = self.schema.index.tolist()
    return self

  def transform_schema(self, X, y=None):
    X = X.copy()

    # Reorder columns
    X = X[self.order]

    # transform schema
    return X.astype(self.schema)
    # inference_schema = self._check_schema(X)
    # if self.schema.keys() == inference_schema.keys():
    # return X.astype(self.schema)
    # else:
    #     if self.schema.keys()!=inference_schema.keys():
    #         raise KeyError(f"There's a missing columns {set(self.schema.keys()).symmetric_difference(set(inference_schema.keys()))}")
    #     elif self.schema.values()!=inference_schema.values():
    #         invalid_schema = [k for k in validate_schema.keys() if validate_schema[k]!=inference_schema[k]]
    #         raise KeyError(f'Invalid schema in features: {", ".join(invalid_schema)}')
    #     else:
    #         raise Exception('Check your data')

  def load_schema(self, schema, X, y=None):
    return X.astype(schema)