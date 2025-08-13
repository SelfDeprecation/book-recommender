import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional

from src.model import NCF
from src.utils import get_device
from src.data_loader import GoodreadsDataset

def train_one_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """Function to train the model for one epoch."""
    model.train()
    total_loss = 0
    # Use leave=False for cleaner progress bars in loops
    progress_bar = tqdm(dataloader, desc="Training", leave=False) 
    for users, items, ratings in progress_bar:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        predictions = model(users, items)
        loss = loss_fn(predictions, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    st_progress_bar=None
):
    """Full model training loop for initial setup."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Average Training Loss: {avg_loss:.4f}")
        if st_progress_bar:
            st_progress_bar.progress((epoch + 1) / num_epochs)

def fine_tune_on_new_user(
    model: NCF,
    new_user_ratings: List[Dict],
    book_map: Dict[int, int],
    device: torch.device
) -> NCF:
    """
    Fine-tunes a new user's embedding based on their ratings.
    """
    model.add_new_user()
    model.to(device)
    new_user_idx = model.user_embedding_gmf.num_embeddings - 1
    
    ratings_df = pd.DataFrame(new_user_ratings)
    ratings_df = ratings_df[ratings_df['book_id'].isin(book_map)]
    if ratings_df.empty:
        print("Warning: New user rated no books that are known to the model. Fine-tuning skipped.")
        return model # Return the expanded model without tuning
        
    ratings_df['book_idx'] = ratings_df['book_id'].map(book_map)
    ratings_df['user_idx'] = new_user_idx
    ratings_df['rating'] = (ratings_df['rating'] - 1.0) / (5.0 - 1.0)
    
    dataset = GoodreadsDataset(ratings_df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    for param in model.parameters():
        param.requires_grad = False
    model.user_embedding_gmf.weight.requires_grad = True
    model.user_embedding_mlp.weight.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.01
    )
    loss_fn = nn.MSELoss()

    print(f"Fine-tuning for new user index {new_user_idx}...")
    model.train()
    for epoch in range(50):
        for users, items, ratings in dataloader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
    
    print("Fine-tuning complete.")
    return model

# --- FULLY CORRECTED FUNCTION ---
def generate_recommendations(
    model: NCF,
    user_id: int,
    user_map: Dict[int, int],
    book_map: Dict[int, int],
    books_df: pd.DataFrame,
    exclude_book_ids: Optional[List[int]] = None,
    top_n: int = 10,
    device: torch.device = get_device()
) -> pd.DataFrame:
    """
    Generates personalized recommendations for a user.

    Args:
        model: The trained/fine-tuned NCF model.
        user_id: The ID of the user.
        user_map: Mapping of user_id -> user_idx.
        book_map: Mapping of book_id -> book_idx.
        books_df: DataFrame with all book information.
        exclude_book_ids: A list of book_ids to exclude from recommendations.
        top_n: The number of recommendations to generate.
        device: The computation device.
        
    Returns:
        A DataFrame with recommended books, or an empty DataFrame if no recommendations can be made.
    """
    model.eval()
    if user_id not in user_map:
        raise ValueError(f"User with ID {user_id} not found in the model's user_map.")
        
    user_idx = user_map[user_id]
    
    if exclude_book_ids is None:
        exclude_book_ids = []
        
    all_book_ids = list(book_map.keys())
    # Identify all books the user has not yet rated/seen.
    unread_book_ids = [book_id for book_id in all_book_ids if book_id not in exclude_book_ids]

    # --- THE KEY FIX: Guard Clause ---
    # If the list of unread books is empty, we cannot proceed.
    # Return an empty DataFrame to avoid a runtime error.
    if not unread_book_ids:
        print("Warning: No unread books found to generate recommendations from.")
        return pd.DataFrame()

    # Prepare data for the model
    user_tensor = torch.tensor([user_idx] * len(unread_book_ids), dtype=torch.long).to(device)
    unread_book_indices = [book_map[book_id] for book_id in unread_book_ids]
    item_tensor = torch.tensor(unread_book_indices, dtype=torch.long).to(device)

    # Get predictions in batches to avoid memory issues if unread_book_ids is large
    recommendations_list = []
    batch_size = 1024 # Process 1024 books at a time
    with torch.no_grad():
        for i in range(0, len(user_tensor), batch_size):
            user_batch = user_tensor[i:i+batch_size]
            item_batch = item_tensor[i:i+batch_size]
            
            predictions = model(user_batch, item_batch)
            
            # Create a temporary DataFrame for the batch results
            batch_recs = pd.DataFrame({
                'book_id': [unread_book_ids[j] for j in range(i, i + len(user_batch))],
                'predicted_rating': predictions.cpu().numpy()
            })
            recommendations_list.append(batch_recs)
    
    # Concatenate all batch results into a single DataFrame
    recommendations = pd.concat(recommendations_list, ignore_index=True)
    
    # Sort by predicted rating and get the top N
    top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(top_n)
    
    # Join with book information to get details like title and author
    rec_details = top_recommendations.merge(books_df, on='book_id')
    
    return rec_details[['title', 'authors', 'average_rating', 'predicted_rating']]
