import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    
    The architecture combines Generalized Matrix Factorization (GMF) and a 
    Multi-Layer Perceptron (MLP) to predict ratings.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int=32, hidden_layers: list=[64, 32, 16]):
        super().__init__()
        
        # --- GMF (Matrix Factorization) Layer ---
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # --- MLP (Multi-Layer Perceptron) Layer ---
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        mlp_layers = []
        input_size = 2 * embedding_dim
        for size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.3))
            input_size = size
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        predict_size = embedding_dim + hidden_layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.item_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.user_embedding_mlp.weight)
        nn.init.xavier_uniform_(self.item_embedding_mlp.weight)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.constant_(self.predict_layer.bias, 0)

    def add_new_user(self):
        """
        Expands the user embedding layers to add a single new user.
        Copies the old weights and initializes a new embedding vector.
        """
        # GMF layer
        old_gmf_weights = self.user_embedding_gmf.weight.data
        new_num_users = self.user_embedding_gmf.num_embeddings + 1
        new_gmf_embedding = nn.Embedding(new_num_users, self.user_embedding_gmf.embedding_dim)
        new_gmf_embedding.weight.data[:new_num_users-1] = old_gmf_weights
        nn.init.xavier_uniform_(new_gmf_embedding.weight.data[-1:])
        self.user_embedding_gmf = new_gmf_embedding

        # MLP layer
        old_mlp_weights = self.user_embedding_mlp.weight.data
        new_mlp_embedding = nn.Embedding(new_num_users, self.user_embedding_mlp.embedding_dim)
        new_mlp_embedding.weight.data[:new_num_users-1] = old_mlp_weights
        nn.init.xavier_uniform_(new_mlp_embedding.weight.data[-1:])
        self.user_embedding_mlp = new_mlp_embedding

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # GMF Path
        user_emb_gmf = self.user_embedding_gmf(users)
        item_emb_gmf = self.item_embedding_gmf(items)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP Path
        user_emb_mlp = self.user_embedding_mlp(users)
        item_emb_mlp = self.item_embedding_mlp(items)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Concatenate the outputs of GMF and MLP
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        
        # Final prediction
        prediction = self.predict_layer(concat_output)
        
        return torch.sigmoid(prediction).squeeze()
