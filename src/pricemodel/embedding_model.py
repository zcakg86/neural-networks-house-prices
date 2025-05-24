#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle
import json

# Function to prepare data
class dataset:
    def __init__(self):
        self.length = None
        self.n_communities = None
        self.year_length = None
        self.week_length = None
        self.scalers = {}
        self.indices = []
        self.community_df = pd.DataFrame()
        self.community_array = np.empty(0)
        self.community_feature_dim = None
        self.community_indices = torch.empty(0)
        self.year_indices = torch.empty(0)
        self.week_indices = torch.empty(0)
        self.property_features = torch.empty(0)
        self.target = torch.empty(0)
        self.dataframe = pd.DataFrame()

    def _prepare_data(self, df):
        """Expected columns: ['sale_date', 'sale_price', 'lat', 'lng', 'sqft', 'sale_nbr','sqft_lot']"""
        # Convert date to datetime
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        # Sort by date
        df = df.sort_values('sale_date')
        # Need to filter out non-null valus
        df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
        # And Zero values
        df = df[df['sale_price'] > 0]
        df = df[df['sqft'] > 0]
        df = df[df['sale_nbr'] > 0]
        df = df[df['sqft_lot'] > 0]
        # Create derived features
        df['price_per_sqft'] = df['sale_price'] / df['sqft']
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.isocalendar().year
        df['week'] = df['sale_date'].dt.isocalendar().week
        df['log_price']= np.log(df['sale_price'])

        self.week_vocab = {value: index for index, value in enumerate(range(1,54))}
        self.year_vocab = {value: index for index, value in enumerate(range(2020,2026))}
        self.week_vocab = {community_id: index for index, community_id in enumerate(range(1,54))}
        def create_vocab(column,min=None,max=None):
            #if min && max:
            ids = sorted(df['community'].unique())
            vocab = {id: index for index, id in enumerate(ids)}
            return vocab
        self.community_vocab = create_vocab('community')
        #community_ids = sorted(df['community'].unique()) # Sort for consistent order across runs
        #self.community_vocab = {community_id: index for index, community_id in enumerate(community_ids)}
        self.n_communities = len(self.community_vocab)
        # Convert community IDs to indices using the vocabulary
        df['community_index'] = df['community'].map(self.community_vocab) # New column with indices
        #df['year_index']
        #df['week_index']
        self.length = df.shape[0]
        self.year_length = len(np.unique(df['year']))
        self.week_length = len(np.unique(df['week']))

        self.dataframe = df
    
    def _get_community_features(self):
        self.community_df = self.dataframe.groupby(['community_index', 'year']).agg({
            'sale_price': ['mean', 'median', 'std'],
            'sqft' : ['mean'],
            'beds' : 'median'
            }).to_dict('index')
        
        # Flatten the dictionary values
        for key, value in self.community_df.items():
            self.community_df[key] = [v for sublist in value.values() for v in (sublist if isinstance(sublist, list) else [sublist])]
        self.community_array = np.array([self.community_df[(c, y)] for c, y in zip(self.dataframe['community_index'], self.dataframe['year'])])
        self.community_feature_dim = self.community_array.shape[1]

    def _processor(self, mode = 'train'):

        for feature in ['sqft','sqft_lot','log_price']: # List the features to scale
            if mode == "train":
                self.scalers[feature] = StandardScaler() # Create a new scaler for each feature
                self.dataframe[f"{feature}_scaled"] = self.scalers[feature].fit_transform(self.dataframe[[feature]]) # Fit and transform
                # Save the scaler
                print(f"{feature}_scaled")
                print(f"{feature}_scaled" in self.dataframe.columns)
                joblib.dump(self.scalers[feature], f"{feature}_scaler.pkl")
            else:  # mode == "val" or "test"
                # Load the pre-fitted scaler
                scaler = joblib.load(f"{feature}_scaler.pkl")
                self.dataframe[feature] = scaler.transform(self.dataframe[[feature]])
                self.dataframe[f"{feature}_scaled"] = self.scalers[feature].transform(self.dataframe[[feature]]) # Fit and transform

        # Community df
        if mode == 'train':
            self.scalers['community'] = StandardScaler()
            self.community_array = self.scalers[feature].fit_transform(self.community_array)
            joblib.dump(self.scalers['community'], "communities_scaler.pkl")

        else: 
            scaler = joblib.load("communities_scaler.pkl")
            self.community_array = scaler.transform(self.community_array)

        # Create tensor with each observation being contiguous, and scale fields.
        self.tensors = TensorDataset(torch.tensor(self.dataframe['community_index'].values, dtype=torch.int),
                                     torch.tensor(self.community_array,dtype = torch.float32),
                                     torch.tensor(self.dataframe['year'].values, dtype=torch.int),
                                     torch.tensor(self.dataframe['week'].values, dtype=torch.int),
                                     torch.tensor(self.dataframe[['sqft_scaled','sqft_lot_scaled']].values, dtype=torch.float32),
                                     torch.tensor(self.dataframe['log_price_scaled'].values, dtype=torch.float32))

class embeddingmodel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, property_dim, 
                 community_embedding_length, community_feature_dim,
                 year_length, week_length):

        # inherit from nn.Module
        super().__init__()
        # Layer dims
        self.property_dim = property_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Embedding dim
        self.community_embedding_length = community_embedding_length
        self.community_feature_dim = community_feature_dim
        self.week_length = week_length
        self.year_length = year_length
        # Embedding Layers
        self.community_embedding = nn.Embedding(int(community_embedding_length), embedding_dim)
        self.year_embedding = nn.Embedding(int(year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(week_length), embedding_dim)

        # Feature Processing Layers
        self.community_feature_layer = nn.Linear(community_feature_dim, hidden_dim)
        self.property_feature_layer = nn.Linear(property_dim, hidden_dim)

        # Calculate combined embedding dimension dynamically
        self.combined_embedding_dim = 3 * embedding_dim

        # Hidden and Output Layers (input_dim calculated dynamically in forward)
        self.hidden_layer1 = nn.Linear(2 * hidden_dim + self.combined_embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, community_indices, community_features, year, week, property_features, targets):
        # Embeddings
        community_embeddings = self.community_embedding(community_indices)
        year_embeddings = self.year_embedding(year)
        week_embeddings = self.week_embedding(week)
        combined_embeddings = torch.cat([community_embeddings, year_embeddings, week_embeddings], dim=-1)

        # Feature Processing
        processed_community_features = self.relu(self.community_feature_layer(community_features))
        processed_property_features = self.relu(self.property_feature_layer(property_features))

        # Combine embeddings and features (calculate input_dim dynamically)
        combined_features = torch.cat([combined_embeddings, processed_community_features, processed_property_features], dim=-1)
        
        embed_dim_attention = combined_features.shape[-1] 
        # Reshape for attention
        combined_features = combined_features.unsqueeze(1)

        # Attention Layer
        attention_layer = nn.MultiheadAttention(embed_dim=embed_dim_attention, num_heads=2, batch_first=True) # Create the layer HERE
        attention_output, _ = attention_layer(combined_features, combined_features, combined_features)
        attention_output = attention_output.squeeze(1)

        # Hidden Layers
        hidden1 = self.relu(self.hidden_layer1(attention_output))  # Use attention_output here
        hidden2 = self.relu(self.hidden_layer2(hidden1))

        # Output Layer
        output = self.output_layer(hidden2)
        print('output shape', output.shape)
        return output, attention_output.shape
class price_predictor:
    def __init__(self, embedding_dim, hidden_dim, property_dim, community_embedding_length,
                 community_feature_dim, year_length, week_length):
        self.device = torch.device('mps' if torch.mps.is_available() 
                                   else 'cuda' if torch.cuda.is_available() 
                                   else 'cpu')
        self.model = embeddingmodel(embedding_dim, hidden_dim, property_dim, 
                                    community_embedding_length, community_feature_dim, 
                                    year_length, week_length).to(self.device)
        # Specify loss measure
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-6)
    def eval(self):
        self.model.eval()

    def train(self, train_loader, val_loader, epochs):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            with torch.autograd.detect_anomaly():
                for batch in train_loader:
                    # Move each tensor in the batch to the device
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the batch
                    community, community_features, year, week, property, targets = batch
                    self.optimizer.zero_grad()
                    print("Community Indices Min:", community.min())
                    print("Community Indices Max:", community.max())
                    print(self.model.community_embedding_length)
                    print("Week Indices Min:", week.min())
                    print("Week Indices Max:", week.max())
                    print(self.model.week_length)
                    print("Size of batch: ", targets.size())
                    predictions, _ = self.model(community, community_features, year,
                                                week, property, targets)
                    if torch.isnan(predictions).any():
                        print("NaN detected in outputs. Skipping this iteration.")
                        continue
                    print("shape of attention output",_)
                    print('predictions shape', predictions.shape)
                    print('predictions')
                    print(predictions.squeeze())
                    print('targets')
                    print(targets)
                    loss = self.criterion(predictions.squeeze(), targets)
                    print('loss')
                    print(loss)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    print(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Move each tensor in the batch to the device
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the batch
                    community, community_features, year, week, property, targets = batch
                    self.optimizer.zero_grad()
                    predictions, _ = self.model(community, community_features, year, week, property, targets)
                    val_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')

        return train_losses, val_losses
    

class modelmanager:
    def __init__(self, dataset, model_name="property_model"):
        self.model = None
        self.dataset = dataset
        self.model_name = model_name
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'metrics': {},
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.embedding_dim = None
        self.hidden_dim = None
        self.property_dim = None
        self.n_communities = self.dataset.n_communities
        self.community_feature_dim = self.dataset.community_feature_dim
        self.week_length = self.dataset.week_length
        self.year_length= self.dataset.year_length

    def train_model(self, embedding_dim, hidden_dim, property_dim, epochs = 10):
        # Split data, and create DataLoader for batces.
        # Sizes from model attributes.
        train_size = int(0.8 * self.dataset.length)
        val_size = self.dataset.length - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset.tensors, [train_size, val_size]
        )

        self.community_embedding_length = train_dataset[:][0].unique().numel()
        # Create year vocabulary for the TRAINING dataset
        train_years = sorted(train_dataset[:][2].unique().tolist())

        self.train_year_vocab = {year: idx for idx, year in enumerate(train_years)}
        self.train_year_vocab["unknown"] = len(self.train_year_vocab)  # Add "unknown" token
        self.train_year_length = len(self.train_year_vocab)

        year_train_tensor = torch.tensor([self.train_year_vocab.get(year.item(),self.train_year_vocab['unknown']) for year in train_dataset.dataset.tensors[2]], dtype=torch.int)
        year_val_tensor = torch.tensor([self.train_year_vocab.get(year.item(),self.train_year_vocab['unknown']) for year in train_dataset.dataset.tensors[2]], dtype=torch.int)

        # Create year vocabulary for the TRAINING dataset
        train_weeks = sorted(train_dataset[:][3].unique().tolist())
        
        self.train_week_vocab = {year: idx for idx, year in enumerate(train_weeks)}
        self.train_week_vocab["unknown"] = len(self.train_week_vocab)  # Add "unknown" token
        self.train_week_length = len(self.train_week_vocab)

        week_train_tensor = torch.tensor([self.train_week_vocab.get(week.item(),self.train_week_vocab['unknown']) for week in train_dataset.dataset.tensors[3]], dtype=torch.int)
        week_val_tensor = torch.tensor([self.train_week_vocab.get(week.item(),self.train_week_vocab['unknown']) for week in train_dataset.dataset.tensors[3]], dtype=torch.int)

        # Create a new TensorDataset with the updated tensors
        new_train_dataset = list(train_dataset.dataset.tensors)  # Convert tuple to list
        new_train_dataset[2] = year_train_tensor  # Replace the old tensor with the updated one
        new_train_dataset[3] = week_train_tensor # Same for weeks

        new_train_dataset = TensorDataset(*new_train_dataset)  # Create new TensorDatase
        train_dataset = Subset(new_train_dataset, train_dataset.indices)  # Use the original indices

        new_val_dataset = list(val_dataset.dataset.tensors)
        new_val_dataset[2] = year_val_tensor  # Replace the old tensor with the updated one
        new_val_dataset[3] = week_val_tensor
        
        new_val_dataset = TensorDataset(*new_val_dataset)  # Create new TensorDataset
        val_dataset = Subset(new_val_dataset, val_dataset.indices)  # Use the original indices


        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        # Create and train model. price_predictor contains model spec.
        predictor = price_predictor(embedding_dim, hidden_dim, property_dim,
                                    self.community_embedding_length, 
                                    self.community_feature_dim,
                                    self.train_year_length,
                                    self.train_week_length)
        
        train_losses, val_losses = predictor.train(train_loader, val_loader, epochs = epochs)

        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses

        self.embedding_dim = embedding_dim, 
        self.hidden_dim = hidden_dim, 
        self.property_dim = property_dim

        self.model = predictor.model

        # Save everything
        #self.save_model()
        # Add predictions to data
        #df_with_pred = manager.add_predictions_to_data(
        #)

        # # Save predictions to CSV
        #df_with_pred.to_csv(f'outputs/results/predictions_{manager.results["timestamp"]}.csv',
        #                    index=False)


    def save_model(self, path="outputs/models/"):
        """Save model, config, processor, and results"""
        import os
        os.makedirs(path, exist_ok=True)
        config = {}
        for name, module in self.model.model.named_modules():
            print(f"Module name: {name}")
            params = {}
            for param_name, param in module.named_parameters(recurse=False):
                print(f"\tParameter name: {param_name}, shape: {param.shape}")
                params[param_name] = [param.shape]
            config[name] = params
        # Save model state
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'model_config': config,
            'results': self.results
        }, f'{path}{self.model_name}_{self.results["timestamp"]}.pth')

        # Save processor (scalers and parameters)
        with open(f'{path}{self.model_name}_{self.results["timestamp"]}_processor.pkl', 'wb') as f:
            pickle.dump(self.processor, f)

        # Save results separately as JSON
        with open(f'{path}{self.model_name}_{self.results["timestamp"]}_results.json', 'w') as f:
            json.dump(self.results, f)

          # Save config as JSON
        with open(f'{path}{self.model_name}_{self.results["timestamp"]}_config.json', 'w') as f:
            json.dump(config, f)

        print(f"Model and results saved in {path}")

# %%
