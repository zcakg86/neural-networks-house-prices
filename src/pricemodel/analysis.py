from src.pricemodel.manager import *
import matplotlib.pyplot as plt
import panel as pn

def analyze_saved_model_features(model_path, processor_path):
    """
    Analyze feature importance from saved model without running new predictions
    """
    # Load checkpoint and processor
    checkpoint = torch.load(model_path, weights_only=False)
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)

    # Get model weights
    model_state = checkpoint['model_state_dict']

    print("Analyzing model weights and architecture...")

    # 1. Analyze LSTM weights (sequence features)
    lstm_weights = model_state['lstm.weight_ih_l0']
    sequence_importance = torch.norm(lstm_weights, dim=0).cpu().numpy()

    # 2. Analyze spatial network weights
    spatial_weights = model_state['spatial_net.0.weight']
    spatial_importance = torch.norm(spatial_weights, dim=0).cpu().numpy()

    # 3. Analyze property network weights
    property_weights = model_state['property_net.0.weight']
    property_importance = torch.norm(property_weights, dim=0).cpu().numpy()

    # 4. Get attention weights if saved in training history
    attention_history = checkpoint.get('results', {}).get('attention_weights', None)
    print(attention_history)
    
    # Create a figure object
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # Plot sequence feature importance
    sequence_features = ['sale_price', 'price_per_sqft', 'local_avg_price']  # adjust based on your features
    axs[0, 0].bar(sequence_features, sequence_importance[:len(sequence_features)])
    axs[0, 0].set_title('Sequence Feature Importance')
    axs[0, 0].set_xticklabels(sequence_features, rotation=45)

    # Plot spatial feature importance
    spatial_features = ['lat', 'lon', 'local_avg_price', 'local_avg_sqft']  # adjust based on your features
    axs[0, 1].bar(spatial_features, spatial_importance[:len(spatial_features)])
    axs[0, 1].set_title('Spatial Feature Importance')
    axs[0, 1].set_xticklabels(spatial_features, rotation=45)

    # Plot property feature importance
    property_features = ['sqft', 'sale_nbr', 'price_per_sqft']  # adjust based on your features
    axs[1, 0].bar(property_features, property_importance[:len(property_features)])
    axs[1, 0].set_title('Property Feature Importance')
    axs[1, 0].set_xticklabels(property_features, rotation=45)

    # Plot training history if available
    if 'results' in checkpoint and 'train_losses' in checkpoint['results']:
        axs[1, 1].plot(checkpoint['results']['train_losses'], label='Train Loss')
        if 'val_losses' in checkpoint['results']:
            axs[1, 1].plot(checkpoint['results']['val_losses'], label='Val Loss')
        axs[1, 1].set_title('Training History')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].legend()
    # Adjust layout
    plt.tight_layout()
    
    pn.pane.Matplotlib(fig, dpi=144).servable()    

    # Create summary dictionary
    importance_summary = {
        'sequence_features': dict(zip(sequence_features, sequence_importance[:len(sequence_features)])),
        'spatial_features': dict(zip(spatial_features, spatial_importance[:len(spatial_features)])),
        'property_features': dict(zip(property_features, property_importance[:len(property_features)]))
    }

    # Print detailed summary
    print("\nFeature Importance Summary:")
    for feature_type, importances in importance_summary.items():
        print(f"\n{feature_type.upper()}:")
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    # Print training metrics if available
    if 'results' in checkpoint and 'metrics' in checkpoint['results']:
        print("\nTraining Metrics:")
        for metric, value in checkpoint['results']['metrics'].items():
            print(f"{metric}: {value}")

    return importance_summary