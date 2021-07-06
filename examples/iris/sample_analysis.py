from ml_report import Report
from sklearn.datasets import load_iris
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

df = load_iris(as_frame=True)["data"]


param_grid = {
    "alpha": [1, 10, 100],
    "l1_ratio": [.25, .5, .75],
}


search = GridSearchCV(
    estimator=ElasticNet(),
    param_grid=param_grid,
    scoring={"r2": "r2"},
    refit="r2", cv=5,
    return_train_score=True,
)


report = Report(
    estimator=ElasticNet(),
    search=search,
    param_grid=param_grid,
)


report.fit(
    df=df,
    iv=["sepal length (cm)", "petal length (cm)", "petal width (cm)"],
    dv="sepal width (cm)",
)

# report.fit(
#     X=df[["sepal length (cm)", "petal length (cm)", "petal width (cm)"]],
#     y=df["sepal width (cm)"],
# )

print(report.iv, report.dv)
print(report.df)

report.build_report()
