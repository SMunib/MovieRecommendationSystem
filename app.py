import pandas as pd
from similarity import compute_similarity_matrix
from predict import make_predictions

# load the ratings
ratings = pd.read_csv("ml-100k/train1.csv")
ratings = ratings.drop(columns="timestamp")
# print(ratings.head())
# print(ratings.isna().sum().sum())

# convert into interaction matrix
interaction_matrix = ratings.pivot(index="userid",columns="itemid",values="rating")
print(interaction_matrix.head())
print(interaction_matrix.isna().sum().sum())

# # compute similarity matrix
# user_similarity_matrix = compute_similarity_matrix(interaction_matrix)
# # print(user_similarity_matrix.head())

# # make predictions and fill the interaction matrix
# new_interaction_matrix = make_predictions(interaction_matrix,user_similarity_matrix)

