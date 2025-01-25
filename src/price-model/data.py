class PropertyPriceModel(nn.Module):
    def __init__(self, sequence_dim, spatial_dim, property_dim, hidden_dim=64):
        super(PropertyPriceModel, self).__init__()
        # Define Layers within model
        # Sequence processing: Similar sales
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Spatial processing
        self.spatial_net = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Property features processing
        self.property_net = nn.Sequential(
            nn.Linear(property_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequences, spatial_features, property_features):
        # Process sequences/temporal features
        lstm_out, _ = self.lstm(sequences)

        # Process spatial and property features
        spatial_out = self.spatial_net(spatial_features)
        property_out = self.property_net(property_features)

        # Calculate attention weights
        spatial_expanded = spatial_out.unsqueeze(1).repeat(1, lstm_out.size(1), 1)
        attention_input = torch.cat([lstm_out, spatial_expanded], dim=2)
        attention_weights = torch.softmax(self.attention(attention_input), dim=1)

        # Apply attention
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)

        # Combine all features
        combined = torch.cat([context_vector, spatial_out, property_out], dim=1)

        # Make prediction
        prediction = self.predictor(combined)

        return prediction, attention_weights
