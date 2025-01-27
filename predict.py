import pandas as pd
import numpy as np

def make_predictions(interaction_matrix, user_similarity_matrix):
    # Create a copy of the interaction matrix to store predicted ratings
    predicted_matrix = interaction_matrix.copy()

    for userid in interaction_matrix.index:
        similar_users = user_similarity_matrix[userid]
        # print(similar_users)

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
    
    # print(predicted_matrix.head())
    return predicted_matrix
