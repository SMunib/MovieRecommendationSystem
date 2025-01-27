from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def compute_similarity_matrix(interaction_matrix):
    interaction_matrix_filled = interaction_matrix.fillna(0)

    user_similarity = cosine_similarity(interaction_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity,index=interaction_matrix.index,columns = interaction_matrix.index)

    return user_similarity_df