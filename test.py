import pandas as pd
import numpy as np
import csv
from sklearn.impute import SimpleImputer
movie = pd.read_csv('old_titles.csv')
median_imputer = SimpleImputer(missing_values= np.nan, strategy = "median")
mean_imputer = SimpleImputer(missing_values= np.nan, strategy = "mean")
movie["imdb_votes"] = np.round(median_imputer.fit_transform(movie[["imdb_votes"]]), 1)
movie["imdb_score"] = np.round(mean_imputer.fit_transform(movie[["imdb_score"]]), 1)
movie["tmdb_popularity"]= np.round(median_imputer.fit_transform(movie[["tmdb_popularity"]]), 1)
movie["tmdb_score"] = np.round(mean_imputer.fit_transform(movie[["tmdb_score"]]),1)

movie.to_csv("titles.csv", index = False)

