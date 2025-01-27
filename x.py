import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def compute_similarity_matrix(interaction_matrix):
    interaction_matrix_filled = interaction_matrix.fillna(0)

    user_similarity = cosine_similarity(interaction_matrix_filled)

    return pd.DataFrame(user_similarity,index=interaction_matrix.index,columns = interaction_matrix.index)

def make_predictions(interaction_matrix, user_similarity_matrix):
    # Create a copy of the interaction matrix to store predicted ratings
    predicted_matrix = interaction_matrix.copy()

    for userid in interaction_matrix.index:
        similar_users = user_similarity_matrix[userid]

        for itemid in interaction_matrix.columns:
            if pd.notna(interaction_matrix.loc[userid, itemid]):
                continue

            ratings_by_similar_users = interaction_matrix[itemid][similar_users.index]
            
            numerator = np.dot(ratings_by_similar_users.fillna(0), similar_users)
            denominator = similar_users[ratings_by_similar_users.notna()].sum()

            if denominator != 0:
                predicted_matrix.loc[userid, itemid] = numerator / denominator
            else:
                predicted_matrix.loc[userid, itemid] = np.nan  # Leave as NaN if no similar users have rated this item
    
    return predicted_matrix

def calculate_mse(predicted,actual):
  mask = ~actual.isna()
  predicted = predicted[mask].values.flatten()
  print(predicted)
  # actual = actual[mask].values.flatten()
  # mse = mean_squared_error(actual,predicted)
  # return mse

def get_top_N_recommendations(predicted,original,n=5):
  top_n_recommendations = {}

  for userid in predicted.index:
    user_ratings = predicted.loc(userid)
    already_rated = original.loc(userid)

    user_ratings = user_ratings[~already_rated]

    top_n_items = user_ratings.nlargest(n).index
    top_n_recommendations[userid] = top_n_items.tolist()
  
  return top_n_recommendations

# load the ratings
ratings = pd.read_csv("traina.csv")
ratings = ratings.drop(columns="timestamp")
ratings.head()

# convert into interaction matrix
interaction_matrix = ratings.pivot(index="userid",columns="itemid",values="rating")
interaction_matrix.head()

# compute similarity matrix
user_similarity_matrix = compute_similarity_matrix(interaction_matrix)
user_similarity_matrix.head()

# make predictions and fill the interaction matrix
predicted_ratings = make_predictions(interaction_matrix,user_similarity_matrix)
predicted_ratings.head()

pred_ratings = predicted_ratings.copy()

pred_ratings = np.round(pred_ratings)
pred_ratings.head()
# print(pred_ratings.isna().sum().sum())

test_data = pd.read_csv("testa.csv")
test_data = test_data.drop(columns=['timestamp'])
test_data.head()

test_interaction_matrix = test_data.pivot(index='userid',columns='itemid',values='rating')
test_interaction_matrix.head()

predicted_for_test = pred_ratings[test_interaction_matrix.notna()]
predicted_for_test.head()

print(pred_ratings.index.equals(test_interaction_matrix.index))  # Should be True
print(pred_ratings.columns.equals(test_interaction_matrix.columns))  # Should be True

mse_test = calculate_mse(predicted_for_test,test_interaction_matrix)
print("Mean Squared Error on test set: ",mse_test)

