import pandas as pd
import polars as pl
import time

path = "dataframe_L03.csv"

start_pd = time.time()
pddf = pd.read_csv(path)
temp_pd = pddf.loc[pddf["person"] > 3].head()
duration_pd = time.time() - start_pd

print("Pandas:", duration_pd) # 0.2706770896911621

start_pl = time.time()
pldf = pl.read_csv(path)
temp_pl = pldf.filter(pldf["person"] > 3).head()
duration_pl = time.time() - start_pl

print("Polars:", duration_pl) # 0.15784287452697754


