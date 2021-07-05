import dill as pickle
import joblib
import json
import numpy as np
import os
import pandas as pd
from eli5 import explain_weights_df
from os.path import dirname, join
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import cross_val_predict, GridSearchCV
from typing import Iterable, Union

from ml_report.report.metrics_report import metrics_report


search_filename = "search.pickle"
model_filename = "model.pickle"


class Report(object):
    def __init__(
        self,
        estimator,
        param_grid,
        df: pd.DataFrame,
        iv: Union[Iterable[str], str],
        dv: Union[Iterable[str], str],
        optimization_metric,
        metrics=None,
        scorers=None,
        search_cv: BaseSearchCV = None,
        rebuild_model=True,
        report_path="ml_reports",
        *args,
        **kwargs,
    ):

        self.estimator = estimator
        self.df = df
        self.iv = iv
        self.dv = dv
        self.param_grid = param_grid

        self.rebuild_model = rebuild_model
        self.report_path = report_path

        self.search_cv = search_cv
        self.y_pred = None

        self.search = None
        self.m = None

        self.metrics_df = None
        self.submetrics_df = None

        self.report_path = report_path
        self._create_report_path()
        self._save_args()

        if search_cv is not None:
            self.search_cv = search_cv
        else:
            self.search_cv = GridSearchCV(
                estimator=self.estimator,
                param_grid=param_grid,
                scoring=scorers,
                # cv=n_splits,
                refit="r2",
                return_train_score=True,
                # n_jobs=n_jobs,
                verbose=10,
            )

    def fit(self, *args, **kwargs):
        self.search_cv.fit(X=self.df[self.iv], y=self.df[self.dv], *args, **kwargs)

    def detailed_report(self, *args, **kwargs):
        pass  # TODO

    def build_report(self, detailed_report=True, save=True, *args, **kwargs):
        df_metrics_report = metrics_report(self.search)

        if save:
            df_metrics_report.to_csv()

    def explain_model(self, save=True, roud=3, *args, **kwargs):
        df_explanation = explain_weights_df(self.m)
        if save:
            df_explanation.round(round).to_csv(self._prepend_report_path("model_explanation.csv"), index=False)
        return df_explanation

    def best_params(self, save=True, *args, **kwargs):
        best_params = self.search.best_params_
        with open(self._prepend_report_path("best_params.json"), "w+") as f:
            json.dump(best_params, f)
        return best_params

    def _create_report_path(self):
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path)

    def _prepend_report_path(self, path):
        return join(self.report_path, path)

    def _save_search(self):
        joblib.dump(self.search_cv, search_filename)
        joblib.dump(self.search_cv.best_estimator_, model_filename)

    def _save_args(self):
        with open(self._prepend_report_path("kwargs.json"), "w+") as f:
            json.dump(self.__dict__, f)


def load_report(report_path):
    with open(join(report_path, "kwargs.json")) as f:
        kwargs = json.load(f)

    report = Report(**kwargs)

    # Load pickled model.
    report.search = joblib.load(search_filename)
    report.m = joblib.load(model_filename)

    return report
