import pandas as pd
import polars as pl
import time

path = "od_dataframe.csv"

start_pd = time.time()
pddf = pd.read_csv(path)
temp_pd = pddf.loc[pddf["Clothing"] > 3].head()
duration_pd = time.time() - start_pd

print("Pandas:", duration_pd) # 4.0461671352386475

start_pl = time.time()
pldf = pl.read_csv(path)
temp_pl = pldf.filter(pldf["Clothing"] > 3).head()
duration_pl = time.time() - start_pl

print("Polars:", duration_pl) # 1.4602329730987549


