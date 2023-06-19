import pandas as pd

ds=pd.read_csv("predictions.csv")

print(ds.head())

print(ds['ABSError'].mean(), ds['ABSMEANError'].mean())