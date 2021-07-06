import dill as pickle
import joblib
import json
import numpy as np
import os
import pandas as pd
from eli5 import explain_weights_df
from os.path import dirname, join
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import cross_val_predict, GridSearchCV
from typing import Iterable, Union

from ml_report.report.metrics_report import metrics_report


_best_params_filename = "best_params.json"
_kwargs_filename = "kwargs.json"
_metrics_report_filename = "metrics.csv"
_model_filename = "model.pickle"
_model_explanation_filename = "model_explanation.csv"
_search_filename = "search.pickle"
_submetrics_report_filename = "submetrics.csv"


class Report(object):
    def __init__(
        self,
        search: BaseSearchCV,
        metrics=None,
        report_path="ml_reports",
        *args,
        **kwargs,
    ):
        self.metrics = metrics

        self.report_path = report_path
        self._create_report_path()

        self.search = search
        self.model = None

        self.df = None
        self.iv = None
        self.dv = None

        self.y_pred = None

    def fit(self, X=None, y=None, df=None, iv=None, dv=None, *args, **kwargs):
        input_Xy = all((X is not None, y is not None))
        input_df = all((df is not None, iv is not None, dv is not None))
        assert input_Xy ^ input_df
        if input_df:
            self.df = df
            self.iv = iv
            self.dv = dv
        if input_Xy:
            # TODO
            self.iv = X.columns

        self.search.fit(X=self.df[self.iv], y=self.df[self.dv], *args, **kwargs)
        self.model = self.search.best_estimator_

    def save_model(self):
        joblib.dump(self.search, _search_filename)
        joblib.dump(self.model, _model_filename)

    def load_model(self):
        self.search = joblib.load(_search_filename)
        self.model = joblib.load(_model_filename)

    def detailed_report(self, *args, **kwargs):
        pass  # TODO

    def build_report(self, detailed_report=True, save=True, *args, **kwargs):
        self.best_params(save=True)
        self.explain_model(save=True)
        self.metrics_report(save=True, round=3)

    def metrics_report(self, save=False, round=3):
        df_metrics_report = metrics_report(self.search)
        if save:
            df_metrics_report.round(round).to_csv(self._prepend_report_path(_metrics_report_filename), index=False)
        return df_metrics_report

    def submetrics_report(self, columns=None, save=False, round=3):
        # TODO:
        pass
        # df_submetrics_report =
        # if save:
        #     df_submetrics_report.round(round).to_csv(self._prepend_report_path(_submetrics_report_filename), index=False)
        # return df_submetrics_report

    def explain_model(self, save=False, round=3, *args, **kwargs):
        df_explanation = explain_weights_df(self.model)
        if save:
            df_explanation.round(round).to_csv(self._prepend_report_path(_model_explanation_filename), index=False)
        return df_explanation

    def best_params(self, save=False, *args, **kwargs):
        best_params = self.search.best_params_
        if save:
            with open(self._prepend_report_path(_best_params_filename), 'w+') as f:
                json.dump(best_params, f)
        return best_params

    def _create_report_path(self):
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path)

    def _prepend_report_path(self, path):
        return join(self.report_path, path)
