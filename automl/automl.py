from .model_evaluation import *
from .feature_engineer import *
from .data_validation import *
from .data_cleansing import *
from .data_encoding import *

from sklearn.compose import make_column_selector as selector
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, \
                                    RandomizedSearchCV, \
                                    GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from tabulate import tabulate
from io import StringIO
import pandas as pd
import numpy as np
import logging
import mlflow


RANDOM_STATE = 101

class AutoML(BaseEstimator, ClassifierMixin):
    def __init__(
        self, preprocessor, model, task='classification', param_grid=None, test_size=0.2, 
        random_state=101, find_best_params=True, feature_selection=None, report=False
        ): 

        """
        Define a AutoML configuration. Here's a pipeline of this function
        Training part.
        1. validate data schema of the training data and save the data schema
        2. split the data into training set and test set to evaluate the model
        3. use ColumnTransformer or ColumnPreprocessor to transform a raw data into a clean data
        4. If find_best_params is True, perform hyperparamter tuning using halving grid search on training dataset to find the best parameters settings
           If find_best_params is False, it will skip this section and only use parameters that define in model
        5. train model and evaluate on test set

        Parameters
        ----------
        preprocessor : ColumnsTransformer or ColumnsPreprocessor
            define a preprocess pipeline   
        model : estimator object
            define a model
        param_grid : None, dict
            dictionary with parameters names as keys and list of parameter settings to try as a values
        test_size : float
            represent the proportion of the dataset to include in the test split
        random_state : int
            controls the shuffling on train_test_split and model configuration
        find_best_params : boolean
            perform hyperparameter tuning if True.
        """
        self.feature_selection = feature_selection
        self.find_best_params = find_best_params
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.param_grid = param_grid
        self.test_size = test_size
        self.best_param = None
        self.model = model
        self.task = task

        self.log_stream = StringIO()
        logging.basicConfig(stream=self.log_stream, level=logging.INFO)
        self.logger = logging.getLogger()
        
    def fit(self, X, y, train_all=True):
        """     
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        train_all : train model with all data (X) if True

        Returns
        -------
        self
        """

        X = X.copy()

        # Validate Schema
        validate_schema = ValidateSchema()
        validate_schema.infer_schema(X)

        # Train Test Split for evaluate model
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=self.test_size, 
                                                            random_state=self.random_state, 
                                                            stratify=y if self.task=='classification' else None)
        self.sample_X_train = X_train.head(10)

        # Preprocessing pipeline
        trans_X_train = self.preprocessor.fit_transform(X_train)
        trans_X_test = self.preprocessor.transform(X_test)
        self.sample_trans_X_train = trans_X_train.head(10)
        
        # Feature_selection
        if feature_selection:
            trans_X_train = self.feature_selection.fit(trans_X_train, y_train)

        # Check if you want to tune hyperparameter
        if self.find_best_params:
            self.tune_model = HalvingGridSearchCV(self.model, self.param_grid)
            self.tune_model.fit(trans_X_train, y_train)
            self.best_param = self.tune_model.best_params_
            self.model = self.model.set_params(**self.best_param)

        # Train model
        self.model.fit(trans_X_train, y_train)

        # evaluate model
        feature_name = trans_X_test.columns.tolist()
        y_pred = self.model.predict(trans_X_test)
        self.result = evaluate_model(y_test, y_pred, 
                                     self.model, 
                                     feature_name,
                                     task=self.task)

        # Fit preprocessor and model with all data available
        if train_all:
            trans_X = self.preprocessor.fit_transform(X)
            if feature_selection:
                trans_X = self.feature_selection.transform(trans_X, y)
            self.model.fit(trans_X, y)

        # Report everything
        if report:
            pass

        return self

    def predict(self, X):

        """
        Validate the input data, preprocess and predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        The input can come through different format, API for real time prediction or
        csv file or pandas dataframe for batch prediction. This predict function need to be
        the same as training data. In case of data type, and 

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        trans_X : transformed input data
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        X = X.copy()

        # Validate Schema
        X = validate_schema.transform_schema(X)

        # Preprocessing pipeline
        trans_X = self.preprocessor.transform(X)

        # predict
        y_pred = self.model.predict(trans_X)
        return trans_X, y_pred

    def to_local(self):
        pass

    def to_mlflow(self, experiment_name, run_name):
        """
        log the training result to mlflow

        Parameters
        ----------
        experiment_name : str
            define experiment name
        run_name : str
            define run name
        """

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:

            # log score
            mlflow.log_metric(key='accuracy', value=self.result['classification_report']['accuracy'])

            # log parameters
            mlflow.log_params(self.model.get_params())

            # log classification report
            mlflow.log_dict(self.result['classification_report'], 'classification_report.json')

            # log confusion matrix
            fig, ax= plot_confusion_matrix(conf_mat=self.result['confusion_matrix'],
                                           show_absolute=True,
                                           show_normed=True)
            mlflow.log_figure(fig, 'confusion_matrix.png')

            # log feature importance
            fig = plt.figure(figsize=(10,6))
            self.result['feature_importance'].plot.barh(fontsize=10)
            mlflow.log_figure(fig, 'feature_importance.png')

            # log file
            mlflow.log_text(self.log_stream.getvalue(), 'log.txt')

            # log hyperparameter tuning result
            result = pd.DataFrame(self.tune_model.cv_results_).sort_values(by='rank_test_score').reset_index(drop=True)
            mlflow.log_text(tabulate(result, headers='keys', tablefmt='fancy_grid'), 'hyperparameter_tuning.txt')

            # log sample raw data
            mlflow.log_text(tabulate(self.sample_X_train, headers='keys', tablefmt='fancy_grid'), 'sample_raw_data.txt')

            # log sample transform data
            mlflow.log_text(tabulate(self.sample_trans_X_train, headers='keys', tablefmt='fancy_grid'), 'sample_transform_data.txt')

            # log data schema
            mlflow.log_dict(self.data_schema, 'data_schema.json')

            # log model
            mlflow.sklearn.log_model(self, 'automl')
