#%%
from src.pricemodel.processor import *
from src.pricemodel.predictor import *
from src.pricemodel.models import *
from datetime import datetime
import torch
import pickle
import json


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
        },
        self.embedding_dim = None, 
        self.hidden_dim = None, 
        self.property_dim = None

    def train_model(self, embedding_dim, hidden_dim, property_dim, epochs = 10):
        # Process data
        ## add later
        # Create and train model. price_predictor contains model spec.
        predictor = price_predictor(dataset, embedding_dim, hidden_dim, property_dim)
        # Create data loaders and train
        train_size = int(0.8 * dataset.length)
        val_size = dataset.length - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        train_losses, val_losses = predictor.train(train_loader, val_loader, epochs = epochs)

        # Initialize model manager
        manager = modelmanager(predictor, dataset)
        manager.results['train_losses'] = train_losses
        manager.results['val_losses'] = val_losses

        manager.embedding_dim = embedding_dim, 
        manager.hidden_dim = hidden_dim, 
        manager.property_dim = property_dim

        # Save everything
        manager.save_model()
        # Add predictions to data
        #df_with_pred = manager.add_predictions_to_data(
        #)

        # # Save predictions to CSV
        #df_with_pred.to_csv(f'outputs/results/predictions_{manager.results["timestamp"]}.csv',
        #                    index=False)
        return manager

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

    def add_predictions_to_data(self, batch_size = 32):
        """Add model predictions to dataframe"""
        self.model.eval()
        predictions = []
        prediction_indices = []

        loader = DataLoader(self.model.dataset.tensors, batch_size)
        
        current_idx = 0

        with torch.no_grad():
            for seq, spat, prop, _, seq_len in loader:
                # Access the underlying model through self.model.model
                # Attach data to device
                pred, _ = self.model.model(
                    seq.to(self.model.device),
                    spat.to(self.model.device),
                    prop.to(self.model.device),
                    seq_len.to(self.model.device)
                )
                # Execute
                batch_predictions = pred.cpu().numpy()
                for i, p in enumerate(batch_predictions):
                    if not np.isnan(p).any():  # Check if prediction was actually made
                        predictions.append(p)
                        prediction_indices.append(current_idx + i)
                
                current_idx += len(seq)
        print(f'Length of predictions {len(predictions)}')
        # Reshape predictions
        predictions = np.array(predictions).reshape(-1, 1)

        # Inverse transform using temporal_scaler
        sequence_shape = sequences.shape
        print(predictions.shape[0])
        dummy_sequence = np.zeros((predictions.shape[0], sequence_shape[2]))
        dummy_sequence[:, 0] = predictions.ravel()  # Put predictions in first column
        predictions = self.processor.scalers['sequences'].inverse_transform(dummy_sequence)[:, 0]
        # Add predictions to dataframe
        # Get existing dataframe
        df_with_pred = df.copy()

        # Initialize predicted_price column with NaN
        df_with_pred['predicted_value'] = pd.NA
        df_with_pred['predicted_price'] = pd.NA

        # Create a mapping of sequence index to original dataframe index
        # This should come from your data preparation step
        if hasattr(self.processor, 'indices'):
            sequence_to_df_idx = self.processor.indices
            
            # Map prediction indices to original dataframe indices
            df_indices = [sequence_to_df_idx[i] for i in prediction_indices]
            
            # Assign predictions to the correct rows
            df_with_pred.iloc[df_indices, df_with_pred.columns.get_loc('predicted_value')] = predictions
            df_with_pred.iloc[df_indices, df_with_pred.columns.get_loc('predicted_price')] = np.exp(predictions)
        else:
            print("Warning: No index mapping found. Using sequential assignment.")
            df_with_pred.iloc[prediction_indices, df_with_pred.columns.get_loc('predicted_value')] = predictions
            df_with_pred.iloc[prediction_indices, df_with_pred.columns.get_loc('predicted_price')] = np.exp(predictions)

        # Add prediction error metrics where we have both actual and predicted prices
        mask = df_with_pred['predicted_price'].notna()
        df_with_pred.loc[mask, 'price_error'] = (
                df_with_pred.loc[mask, 'predicted_price'] -
                df_with_pred.loc[mask, 'sale_price']
        )
        df_with_pred.loc[mask, 'price_error_pct'] = (
                df_with_pred.loc[mask, 'price_error'] /
                df_with_pred.loc[mask, 'sale_price'] * 100
        )

        # Print some debugging information
        print(f"Original dataframe shape: {df.shape}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of non-null predictions: {df_with_pred['predicted_price'].notna().sum()}")
        print(f"Mean Absolute Percentage Error: {df_with_pred['price_error_pct'].abs().mean()}")

        return df_with_pred

    def analyze_results(self, df_with_pred):
        """Analyze prediction results"""
        # Calculate metrics
        metrics = {
            'mae': np.mean(np.abs(df_with_pred['price_error'].dropna())),
            'mape': np.mean(np.abs(df_with_pred['price_error_pct'].dropna())),
            'rmse': np.sqrt(np.mean(np.square(df_with_pred['price_error'].dropna())))
        }

        # Store metrics
        self.results['metrics'] = metrics


        return metrics


# Usage example
def train_and_save_model(df, sequence_length=12, epochs = 10):
    # Process data
    processor = data_processor(sequence_length=sequence_length)
    sequences, spatial_features, property_features, targets, df, sequence_lengths = processor.prepare_data(df)

    # Create and train model
    predictor = price_predictor(
        sequence_dim=sequences.shape[2],
        spatial_dim=spatial_features.shape[1],
        property_dim=property_features.shape[1]
    )
    # Create data loaders and train
    dataset = maketensor(sequences, spatial_features, property_features, targets, sequence_lengths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    train_losses, val_losses = predictor.train(train_loader, val_loader, epochs = epochs)

    # Initialize model manager
    manager = modelmanager(predictor, processor)
    manager.results['train_losses'] = train_losses
    manager.results['val_losses'] = val_losses

    # Save everything
    manager.save_model()
    # Add predictions to data
    df_with_pred = manager.add_predictions_to_data(
         df, sequences, spatial_features, property_features, sequence_lengths
    )

    # # Save predictions to CSV
    df_with_pred.to_csv(f'outputs/results/predictions_{manager.results["timestamp"]}.csv',
                        index=False)
    return manager

def load_saved_model_with_config(path_prefix):
    """
    Load saved model with configuration
    """
    # Load configuration
    with open(f'{path_prefix}_config.json', 'r') as f:
        config = json.load(f)

    # Load processor
    with open(f'{path_prefix}_processor.pkl', 'rb') as f:
        processor = pickle.load(f)

    # Initialize model with saved configuration
    predictor = price_predictor(
        sequence_dim=config['sequence_dim'],
        spatial_dim=config['spatial_dim'],
        property_dim=config['property_dim'],
        hidden_dim=config['hidden_dim']
    )

    # Load state dictionaries
    checkpoint = torch.load(f'{path_prefix}.pth')
    predictor.model.load_state_dict(checkpoint['model_state_dict'])
    predictor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return predictor, processor

# %%
