from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

# class DropMostNA(BaseEstimator, TransformerMixin):
#   def __init__(self, na_threshold):
#     self.na_threshold = na_threshold
  
#   def _check_most_na_columns(self, df, na_threshold):
#     count_na = df.isna().mean()
#     most_na_columns = count_na[count_na>na_threshold].index
#     return most_na_columns
  
#   def fit(self, X, y=None):
#     X = X.copy()
#     self.most_na_columns = self._check_most_na_columns(X, self.na_threshold)
#     return self
  
#   def transform(self, X, y=None):
#     X = X.copy()
#     X = X.drop(self.most_na_columns, axis=1)
#     return X

# class CleanData(BaseEstimator, TransformerMixin):
#   def __init__(self, fill_numerical, fill_categorical, num_value=None, cat_value=None):
#     '''
#     Constructs all the necessary attribute

#       Parameters:
#         fill_numerical (str): 
#           Method for impute numerical data (mean, mode, median and constant)
#         fill_categorical (str): 
#           Method for impute categorical data (mode and constant)
#         num_value (int): 
#           Define constant value if fill_numerical is constant
#         cat_value (str): 
#           Define constant value if fill_categorical is constant
#     '''
#     self.fill_numerical = fill_numerical
#     self.fill_categorical = fill_categorical
#     self.num_value = num_value
#     self.cat_value = cat_value
#     self.cleaner = {}

#   def fit(self, X, y=None):
#     X = X.copy()
#     self.num_features = X.select_dtypes(include='number').columns
#     self.cat_features = X.select_dtypes(exclude='number').columns

#     # for numerical columns
#     if self.fill_numerical == 'mean':
#       self.cleaner.update(X[self.num_features].mean().to_dict())
#     elif self.fill_numerical == 'mode':
#       self.cleaner.update(X[self.num_features].mode().iloc[0].to_dict())
#     elif self.fill_numerical == 'median':
#       self.cleaner.update(X[self.num_features].median().to_dict())
#     elif self.fill_numerical == 'constant':
#       self.cleaner.update(dict(zip(self.num_features, 
#                                    [self.num_value]*len(self.num_features))
#       ))
#     # for categorical columns
#     if self.fill_categorical == 'mode':
#       self.cleaner.update(X[self.cat_features].mode().iloc[0].to_dict())
#     elif self.fill_categorical == 'constant':
#       self.cleaner.update(dict(zip(self.cat_features, 
#                                    [self.cat_value]*len(self.cat_features))
#       ))
#     return self

#   def transform(self, X, y=None):
#     X = X.copy()
#     X = X.fillna(self.cleaner)
#     return X