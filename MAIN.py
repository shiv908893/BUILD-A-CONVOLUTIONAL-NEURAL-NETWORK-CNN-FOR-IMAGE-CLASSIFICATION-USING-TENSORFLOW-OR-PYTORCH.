import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. SETUP & DATA PREPARATION
# ---------------------------------------------------------
# Generate synthetic data: 10 Users x 10 Movies
np.random.seed(42)
users = [f'User_{i}' for i in range(1, 11)]
movies = [f'Movie_{i}' for i in range(1, 11)]

# Create sparse matrix (30% missing values)
data = np.random.randint(1, 6, size=(10, 10)).astype(float)
data[np.random.rand(10, 10) < 0.3] = np.nan
df_pivot = pd.DataFrame(data, index=users, columns=movies)

print("Sparse User-Item Rating Matrix (Top 5):")
print(df_pivot.head())

# ---------------------------------------------------------
# 2. COLLABORATIVE FILTERING (USER-BASED)
# ---------------------------------------------------------
# We find users similar to each other using Cosine Similarity
df_filled = df_pivot.fillna(0)
user_sim = cosine_similarity(df_filled)
user_sim_df = pd.DataFrame(user_sim, index=users, columns=users)

# Logic: Predict a rating by taking a weighted average of similar users' ratings
def predict_rating(user, movie, pivot_df, sim_df):
    if not np.isnan(pivot_df.loc[user, movie]):
        return pivot_df.loc[user, movie]
    
    similar_users = sim_df[user].drop(user)
    neighbor_ratings = pivot_df.loc[similar_users.index, movie].fillna(0)
    
    # Weighted average calculation
    score = np.dot(similar_users, neighbor_ratings) / (np.sum(np.abs(similar_users)) + 1e-9)
    return score

# ---------------------------------------------------------
# 3. MATRIX FACTORIZATION (SVD)
# ---------------------------------------------------------
# Step: Decompose R into U, Sigma, and Vt matrices
user_ratings_mean = df_pivot.mean(axis=1)
df_demeaned = df_pivot.sub(user_ratings_mean, axis=0).fillna(0)

# Singular Value Decomposition
U, sigma, Vt = np.linalg.svd(df_demeaned, full_matrices=False)
sigma = np.diag(sigma)

# Reconstruct the matrix to fill in the gaps
all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
df_predicted = pd.DataFrame(all_predicted_ratings, index=users, columns=movies)

# ---------------------------------------------------------
# 4. EVALUATION & RESULTS
# ---------------------------------------------------------
# Root Mean Squared Error (RMSE) on known ratings
mask = ~np.isnan(df_pivot.values)
rmse = np.sqrt(mean_squared_error(df_pivot.values[mask], all_predicted_ratings[mask]))

print(f"\nModel Performance (RMSE): {rmse:.4f}")
print("\nPredicted Ratings for User_1 (First 5 Movies):")
print(df_predicted.loc['User_1'].head())

# Visualize the original sparsity
plt.figure(figsize=(8, 6))
sns.heatmap(df_pivot, annot=True, cmap='YlGnBu', cbar_klabels={'label': 'Rating'})
plt.title('Original Rating Matrix (with missing values)')
plt.show()s
