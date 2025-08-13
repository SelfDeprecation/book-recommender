import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

class GoodreadsDataset(Dataset):
    """PyTorch Dataset for book ratings."""
    def __init__(self, ratings: pd.DataFrame):
        self.users = torch.tensor(ratings['user_idx'].values, dtype=torch.long)
        self.books = torch.tensor(ratings['book_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.books[idx], self.ratings[idx]

def preprocess_data(
    data_path: Path,
    min_ratings_user: int = 5,
    min_ratings_book: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Loads, filters, and preprocesses data about books and ratings for initial training.
    Note: The logic for handling new users has been removed from this function.
    """
    print("Loading initial data...")
    ratings = pd.read_csv(data_path / "ratings.csv")
    books = pd.read_csv(data_path / "books.csv")

    print("Filtering data...")
    user_counts = ratings['user_id'].value_counts()
    book_counts = ratings['book_id'].value_counts()
    
    active_users = user_counts[user_counts >= min_ratings_user].index
    active_books = book_counts[book_counts >= min_ratings_book].index
    
    ratings_filtered = ratings[
        ratings['user_id'].isin(active_users) & ratings['book_id'].isin(active_books)
    ]
    
    print(f"{len(ratings_filtered)} ratings remaining after filtering.")

    # Create mappings for embeddings
    unique_users = ratings_filtered['user_id'].unique()
    unique_books = ratings_filtered['book_id'].unique()
    
    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    book_map = {book_id: i for i, book_id in enumerate(unique_books)}
    
    ratings_filtered['user_idx'] = ratings_filtered['user_id'].map(user_map)
    ratings_filtered['book_idx'] = ratings_filtered['book_id'].map(book_map)
    
    # Normalize ratings to the [0, 1] range for better training
    min_rating, max_rating = ratings_filtered['rating'].min(), ratings_filtered['rating'].max()
    ratings_filtered['rating'] = (ratings_filtered['rating'] - min_rating) / (max_rating - min_rating)

    # Keep only the books in books_df that are present in the filtered ratings
    books_df_filtered = books[books['book_id'].isin(book_map.keys())].copy()
    
    books_df_filtered['book_idx'] = books_df_filtered['book_id'].map(book_map)
    
    return ratings_filtered, books_df_filtered, user_map, book_map


def create_dataloaders(
    ratings_df: pd.DataFrame, batch_size: int = 1024
) -> Tuple[DataLoader, DataLoader]:
    """Creates DataLoaders for training and validation."""
    train_df, val_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
    
    train_dataset = GoodreadsDataset(train_df)
    val_dataset = GoodreadsDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
