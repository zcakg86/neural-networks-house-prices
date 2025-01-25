class modelmanager:
    def __init__(self, model, processor, model_name="property_model"):
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'metrics': {},
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    def save_model(self, path="outputs/models/"):
        """Save model, config, processor, and results"""
        import os
        os.makedirs(path, exist_ok=True)

        config = {
              'sequence_dim': self.model.model.lstm.input_size,
              'spatial_dim': next(self.model.model.spatial_net.parameters()).shape[1],
              'property_dim': next(self.model.model.property_net.parameters()).shape[1],
              'hidden_dim': self.model.model.lstm.hidden_size
        }
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

    def add_predictions_to_data(self, df, sequences, spatial_features, property_features):
        """Add model predictions to dataframe"""
        self.model.eval()
        predictions = []

        # Create DataLoader for prediction
        dataset = maketensor(sequences, spatial_features, property_features,
                                  np.zeros(len(sequences)))  # dummy targets
        loader = DataLoader(dataset, batch_size=32)

        with torch.no_grad():
            for seq, spat, prop, _ in loader:
                # Access the underlying model through self.model.model
                pred, _ = self.model.model(
                    seq.to(self.model.device),
                    spat.to(self.model.device),
                    prop.to(self.model.device)
                )
                predictions.extend(pred.cpu().numpy())

        # Reshape predictions
        predictions = np.array(predictions).reshape(-1, 1)

        # Inverse transform using temporal_scaler
        sequence_shape = sequences.shape
        dummy_sequence = np.zeros((predictions.shape[0], sequence_shape[2]))
        dummy_sequence[:, 0] = predictions.ravel()  # Put predictions in first column
        predictions = self.processor.scalers['temporal'].inverse_transform(dummy_sequence)[:, 0]

        # Add predictions to dataframe
        df_with_pred = df.copy()

        # Initialize predicted_price column with NaN
        df_with_pred['predicted_price'] = pd.NA

        # Calculate the correct indices for predictions
        start_idx = self.processor.sequence_length
        end_idx = start_idx + len(predictions)

        # Assign predictions to the correct rows
        df_with_pred.loc[df_with_pred.index[start_idx:end_idx], 'predicted_price'] = predictions

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
        print(f"Predictions start index: {start_idx}")
        print(f"Predictions end index: {end_idx}")
        print(f"Number of non-null predictions: {df_with_pred['predicted_price'].notna().sum()}")

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
    sequences, spatial_features, property_features, targets, df = processor.prepare_data(df)

    # Create and train model
    predictor = predictor(
        sequence_dim=sequences.shape[2],
        spatial_dim=spatial_features.shape[1],
        property_dim=property_features.shape[1]
    )
    # Create data loaders and train
    dataset = maketensor(sequences, spatial_features, property_features, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    train_losses, val_losses = predictor.train(train_loader, val_loader, epochs = epochs)

    # Initialize model manager
    manager = modelmanager(predictor, processor)
    manager.results['train_losses'] = train_losses
    manager.results['val_losses'] = val_losses

    # Add predictions to data
    df_with_pred = manager.add_predictions_to_data(
        df, sequences, spatial_features, property_features
    )

    # Analyze results
    metrics = manager.analyze_results(df_with_pred)

    # Save everything
    manager.save_model()


    # Save predictions to CSV
    df_with_pred.to_csv(f'outputs/results/predictions_{manager.results["timestamp"]}.csv',
                        index=False)

    return manager, df_with_pred

