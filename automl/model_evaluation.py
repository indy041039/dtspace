from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, \
                            accuracy_score, recall_score, precision_score, \
                            mean_squared_error, r2_score, f1_score

def evaluate_model(y_test, y_pred, model, feature_names, task = 'classification'):
    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):
      feature_importance = pd.Series(model.feature_importances_, 
                                    index=feature_names) \
                                    .sort_values(ascending=True)
    elif hasattr(model, 'coef_'):
      feature_importance = pd.Series(model.coef_[0], 
                                  index=feature_names) \
                                  .sort_values(ascending=True)   
    else:
      feature_importance = None

    # Evaluate model          
    if task == 'classification':
      cls_report = classification_report(y_test, y_pred, output_dict=True)
      conf_matrix = confusion_matrix(y_test, y_pred)
      return {
        'classification_report': cls_report,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test,y_pred),
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
      }

    elif task == 'regression':
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2 = r2_score(y_test, y_pred) 
      return {
        'rmse': rmse,
        'r2_score': r2,
        'feature_importance': feature_importance
      }