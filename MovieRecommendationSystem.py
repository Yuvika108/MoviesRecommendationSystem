import pandas as pd
import numpy as np

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(movies.head())
print(credits.head())
