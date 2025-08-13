import streamlit as st
import pandas as pd
from pathlib import Path
import time

from src.data_loader import preprocess_data, create_dataloaders
from src.model import NCF
from src.engine import train_model, generate_recommendations, fine_tune_on_new_user
from src.utils import get_device, save_artifacts, load_artifacts

# --- Page Configuration ---
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# --- Global Variables and State ---
DATA_PATH = Path("data")
ARTIFACTS_PATH = Path("artifacts")
MODEL_FILE = ARTIFACTS_PATH / "model.pt"
DEVICE = get_device()
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32

if 'app_ready' not in st.session_state:
    st.session_state.app_ready = False
if 'new_user_ratings' not in st.session_state:
    st.session_state.new_user_ratings = []


# --- UI Functions (no changes here) ---
@st.cache_data
def load_base_books():
    books_df = pd.read_csv(DATA_PATH / "books.csv")
    books_df['display'] = books_df['title'].str.slice(0, 70) + " - " + books_df['authors'].str.slice(0, 30)
    return books_df

def handle_initial_training():
    st.info("The model has not been trained yet. Starting the process...")
    with st.spinner("Step 1/3: Preprocessing data... This may take a few minutes."):
        ratings_df, books_df_processed, user_map, book_map = preprocess_data(DATA_PATH)
        st.session_state.ratings_df = ratings_df
        st.session_state.books_df_processed = books_df_processed
        st.session_state.user_map = user_map
        st.session_state.book_map = book_map
    with st.spinner("Step 2/3: Creating DataLoaders..."):
        train_loader, _ = create_dataloaders(ratings_df, batch_size=2048)
    model = NCF(
        num_users=len(user_map),
        num_items=len(book_map),
        embedding_dim=EMBEDDING_DIM
    )
    st.write("Step 3/3: Training the model...")
    progress_bar = st.progress(0)
    start_time = time.time()
    train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE, progress_bar)
    end_time = time.time()
    st.success(f"Model successfully trained in {end_time - start_time:.2f} seconds!")
    with st.spinner("Saving artifacts..."):
        save_artifacts(model, user_map, book_map, books_df_processed, ARTIFACTS_PATH)
    st.session_state.model = model
    st.session_state.app_ready = True
    st.rerun()

def handle_artifacts_loading():
    with st.spinner("Loading artifacts..."):
        model_state, user_map, book_map, books_df_processed = load_artifacts(ARTIFACTS_PATH)
        model = NCF(len(user_map), len(book_map), EMBEDDING_DIM)
        model.load_state_dict(model_state)
        st.session_state.model = model
        st.session_state.user_map = user_map
        st.session_state.book_map = book_map
        st.session_state.books_df_processed = books_df_processed
        st.session_state.ratings_df = pd.read_csv(DATA_PATH / "ratings.csv")
    st.success("Trained model and data loaded successfully!")
    st.session_state.app_ready = True
    st.rerun()


# --- Main UI ---
if not st.session_state.app_ready:
    st.title("📚 Personalized Book Recommender System")
    st.markdown("Welcome! This system will help you find new and interesting books based on your preferences.")
    if MODEL_FILE.exists():
        st.info("A pre-trained model was found.")
        if st.button("Load Trained Model", type="primary"):
            handle_artifacts_loading()
    else:
        st.warning("Pre-trained model not found.")
        if st.button("Start Initial Model Training", type="primary"):
            handle_initial_training()
else:
    st.sidebar.success("Model is ready to use!")
    st.header("Rate a few books to get personalized recommendations")
    base_books = load_base_books()
    
    selected_book_info = st.multiselect(
        "Start typing a book title to find it in the list:",
        options=base_books['display'].tolist(),
        key="book_selector"
    )

    if selected_book_info:
        current_ratings = []
        for book_display in selected_book_info:
            book_id = base_books[base_books['display'] == book_display].iloc[0]['book_id']
            title = base_books[base_books['display'] == book_display].iloc[0]['title']
            rating = st.slider(f"Your rating for '{title}':", 1, 5, 3, key=f"rating_{book_id}")
            current_ratings.append({'book_id': int(book_id), 'rating': rating})
        st.session_state.new_user_ratings = current_ratings

    if st.button("Get Recommendations", disabled=len(st.session_state.new_user_ratings) < 3):
        if len(st.session_state.new_user_ratings) < 3:
            st.error("Please rate at least 3 books to get quality recommendations.")
        else:
            with st.spinner('Recommendation magic in progress! Adapting the model for you... ✨'):
                # 1. Create a fresh instance of the model with the original dimensions.
                model_to_finetune = NCF(
                    num_users=len(st.session_state.user_map),
                    num_items=len(st.session_state.book_map),
                    embedding_dim=EMBEDDING_DIM
                )
                # 2. Load the trained weights from our base model.
                model_to_finetune.load_state_dict(st.session_state.model.state_dict())
                
                # 3. Now, fine-tune this clean, independent model instance.
                fine_tuned_model = fine_tune_on_new_user(
                    model=model_to_finetune,
                    new_user_ratings=st.session_state.new_user_ratings,
                    book_map=st.session_state.book_map,
                    device=DEVICE
                )
                
                # The rest of the logic remains the same
                new_user_id = max(st.session_state.user_map.keys()) + 1
                updated_user_map = st.session_state.user_map.copy()
                updated_user_map[new_user_id] = fine_tuned_model.user_embedding_gmf.num_embeddings - 1
                
                rated_book_ids = [r['book_id'] for r in st.session_state.new_user_ratings]

                recommendations = generate_recommendations(
                    model=fine_tuned_model,
                    user_id=new_user_id,
                    user_map=updated_user_map,
                    book_map=st.session_state.book_map,
                    books_df=st.session_state.books_df_processed,
                    exclude_book_ids=rated_book_ids,
                    top_n=10,
                    device=DEVICE
                )

            st.header("Your Personalized Recommendations:")
            if not recommendations.empty:
                st.dataframe(
                    recommendations,
                    column_config={
                        'title': st.column_config.TextColumn("Title", width="large"),
                        'authors': st.column_config.TextColumn("Author"),
                        'average_rating': st.column_config.NumberColumn("Avg. Rating", format="%.2f"),
                        'predicted_rating': st.column_config.ProgressColumn(
                            "Your Predicted Score", format="%.2f", min_value=0, max_value=1,
                        ),
                    },
                    hide_index=True
                )
            else:
                st.warning("Unfortunately, we couldn't generate recommendations. Try rating different books.")

st.sidebar.info(
    """
    **How It Works**
    1.  **Train/Load**: A base model is trained on 1M ratings or loaded from disk.
    2.  **Your Ratings**: You select and rate books you have read.
    3.  **Fine-Tuning**: A new embedding vector is created for you. The model is then **rapidly fine-tuned** (in seconds) on just your ratings to learn your unique taste.
    4.  **Recommendations**: The adapted model predicts which *new* books you are most likely to enjoy, explicitly excluding those you just rated.
    """
)
