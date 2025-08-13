import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import pandas as pd

def get_device() -> torch.device:
    """Determines the available device (GPU or CPU) for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_artifacts(
    model: torch.nn.Module,
    user_map: Dict[int, int],
    book_map: Dict[int, int],
    books_df: pd.DataFrame,
    path: Path = Path("artifacts"),
):
    """
    Saves the trained model and user/book mappings.
    
    Args:
        model (torch.nn.Module): The trained model.
        user_map (Dict[int, int]): Dictionary for mapping user_id to index.
        book_map (Dict[int, int]): Dictionary for mapping book_id to index.
        books_df (pd.DataFrame): DataFrame with book information.
        path (Path): Path to save the artifacts.
    """
    path.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), path / "model.pt")
    
    # Save mappings and the books DataFrame
    with open(path / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(path / "book_map.pkl", "wb") as f:
        pickle.dump(book_map, f)
        
    books_df.to_csv(path / "books_df.csv", index=False)
    
    print(f"Artifacts saved to {path}")


def load_artifacts(
    path: Path = Path("artifacts"),
) -> Tuple[Dict, Dict[int, int], Dict[int, int], Any]:
    """
    Loads the saved model and mappings.

    Args:
        path (Path): Path to the artifacts folder.

    Returns:
        Tuple: Model state dictionary, user mapping, book mapping, books DataFrame.
    """
    
    # Load model state dictionary
    model_state_dict = torch.load(path / "model.pt", map_location="cpu")
    
    # Load mappings
    with open(path / "user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open(path / "book_map.pkl", "rb") as f:
        book_map = pickle.load(f)
        
    books_df = pd.read_csv(path / "books_df.csv")
    
    print(f"Artifacts loaded from {path}")
    return model_state_dict, user_map, book_map, books_df
