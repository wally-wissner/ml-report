from sklearn.datasets import load_iris


ds = load_iris(as_frame=True)
df = ds["data"]
print(df)
