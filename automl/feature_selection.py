from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np

def drop_column(X, col, drop_most_na=False, na_threshold=0.2):
    '''
    Drop selected columns and high nan columns
    '''
    drop_X = X.drop(col, axis=1)
    if drop_most_na:
        count_nan = X.isna().mean()
        most_na_col = count_nan[count_nan > na_threshold].index.tolist()
        drop_X = drop_X.drop(most_na_col, axis=1)
    return drop_X

class ColumnSelector(BaseEstimator, SelectorMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        self._n_features = X.shape[1]
        return self

    def _get_support_mask(self):
        check_is_fitted(self, '_n_features')
        if self.cols is None:
            mask = np.ones(self._n_features, dtype=bool)
        else:
            mask = np.zeros(self._n_features, dtype=bool)
            mask[list(self.cols)] = True
        return mask

class FeatureSelection(BaseEstimator, TransformerMixin):
  def __init__(self, selector):
    self.selector = selector

  def _get_columns(self, feature_name):
    for s in self.selector:
      feature_name = feature_name[s.get_support()]
    return feature_name

  def fit(self, X, y=None):
    self.selector.fit(X, y)
    self.feature_name = self._get_columns(X.columns)
    return self

  def transform(self, X, y=None):
    X = pd.DataFrame(self.selector.transform(X), columns=self.feature_name)
    return X
        


