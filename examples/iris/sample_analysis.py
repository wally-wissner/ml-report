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
    search=GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, scoring="r2"),
    param_grid=param_grid,
)
report.fit(
    df=df,
    iv=["sepal length (cm)", "sepal width (cm)", "petal length (cm)"],
    dv="petal width (cm)",
)
report.build_report()
