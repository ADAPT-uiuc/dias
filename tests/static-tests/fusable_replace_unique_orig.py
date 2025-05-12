df['target_ages'] = df['rating'].replace(ratings_ages)
df['target_ages'].unique()