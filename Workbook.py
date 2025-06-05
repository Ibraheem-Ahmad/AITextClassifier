import pandas as pd


#df = pd.read_csv("C:\MasterFolder\Coding\School\INFO Project\data.csv").sample(n=100000, random_state=42)

df = pd.read_csv("smalldata.csv")
df.to_parquet("smalldata.parquet", index=False)
