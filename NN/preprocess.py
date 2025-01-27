import pandas as pd
import numpy as np

data_folder = 'ml-100k/'

# Load the required files
ua_base = pd.read_csv(data_folder + 'ua.base', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
ua_test = pd.read_csv(data_folder + 'ua.test', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u_user = pd.read_csv(data_folder + 'u.user', sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip'])

u_user['gender'] = u_user['gender'].apply(lambda x: 1 if x == 'M' else 0)

u_item = pd.read_csv(data_folder + 'u.item', sep='|', header=None, 
                     names=['item_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(1, 20)],
                     encoding='ISO-8859-1')
u_item = u_item.drop(columns=['movie_title', 'release_date', 'video_release_date', 'imdb_url'])

merged_base = pd.merge(ua_base, u_item, on='item_id')
merged_test = pd.merge(ua_test, u_item, on='item_id')

def compute_user_profile(user_ratings):
    profile = np.zeros(19)
    total_ratings = 0
    for _, row in user_ratings.iterrows():
        rating = row['rating']
        genres = row[4:].values
        weighted_genres = genres * rating
        profile += weighted_genres
        total_ratings += rating
    if total_ratings > 0:
        profile = profile / total_ratings
    return profile

user_profiles = {}
for user_id in merged_base['user_id'].unique():
    user_ratings = merged_base[merged_base['user_id'] == user_id]
    user_profiles[user_id] = compute_user_profile(user_ratings)

user_train_data = []
for _, row in merged_base.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    age = u_user[u_user['user_id'] == user_id]['age'].values[0]
    gender = u_user[u_user['user_id'] == user_id]['gender'].values[0]
    user_profile = user_profiles[user_id]
    interaction = 1 if row['rating'] >= 3 else 0
    item_genres = row[4:].values
    user_train_data.append([user_id, item_id, age, gender] + list(user_profile) + list(item_genres) + [interaction])

user_train_df = pd.DataFrame(user_train_data, columns=['user_id', 'item_id', 'age', 'gender'] + [f'genre_{i}' for i in range(1, 20)] + [f'item_genre_{i}' for i in range(1, 20)] + ['interaction'])
user_train_df.to_csv('user_train.csv', index=False)

user_test_data = []
for _, row in merged_test.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    age = u_user[u_user['user_id'] == user_id]['age'].values[0]
    gender = u_user[u_user['user_id'] == user_id]['gender'].values[0]
    user_profile = user_profiles[user_id]
    interaction = 1 if row['rating'] >= 3 else 0
    item_genres = row[4:].values
    user_test_data.append([user_id, item_id, age, gender] + list(user_profile) + list(item_genres) + [interaction])

user_test_df = pd.DataFrame(user_test_data, columns=['user_id', 'item_id', 'age', 'gender'] + [f'genre_{i}' for i in range(1, 20)] + [f'item_genre_{i}' for i in range(1, 20)] + ['interaction'])
user_test_df.to_csv('user_test.csv', index=False)

print("CSV files created: user_train.csv, user_test.csv")
