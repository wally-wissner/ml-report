# Purpose

**ml-report** is a Python module providing in-depth metrics and diagnostics for training and evaluating Machine Learning models.


# Installation

    pip install ml-report


# Usage



# Examples

    from ml_report import Report
    from sklearn.datasets import load_iris
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    
    df = load_iris(as_frame=True)["data"]
    param_grid = {
        "alpha": [1, 10, 100],
        "l1_ratio": [.25, .5, .75],
    }
    report = Report(
        estimator=ElasticNet(),
        search=GridSearchCV(
            estimator=ElasticNet(),
            param_grid=param_grid,
            scoring={"r2": "r2"},
            refit="r2", cv=5,
            return_train_score=True,
        ),
        param_grid=param_grid,
    )
    report.fit(
        df=df,
        iv=["sepal length (cm)", "petal length (cm)", "petal width (cm)"],
        dv="sepal width (cm)",
    )
    report.build_report()
