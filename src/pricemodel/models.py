import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_forecasting.models.nn.rnn import LSTM

class initialmodel(nn.Module):
    def __init__(self, sequence_dim, spatial_dim, property_dim, hidden_dim=64):
        super(initialmodel, self).__init__()
        # Define Layers within model
        # Sequence processing: Similar sales
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Community embedding
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
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequences, spatial_features, property_features, sequence_lengths):
        
        lstm_out_placeholder = torch.zeros(sequences.shape[0],sequences.shape[1],self.lstm.hidden_size)
        
        # Filter out non zero sequences with Mask
        non_zero_mask = sequence_lengths > 0
        filtered_sequences = sequences[non_zero_mask]
        filtered_lengths = sequence_lengths[non_zero_mask]

        filtered_lengths, perm_idx = filtered_lengths.sort(0, descending=True)
        filtered_sequences = sequences[perm_idx]
        print(f'Number of non zero sequences: {len(filtered_sequences)}')
        if len(filtered_sequences) > 0:
        # Step 2: Pack the sequences
            packed_sequences = pack_padded_sequence(filtered_sequences, filtered_lengths.cpu(), batch_first=True, enforce_sorted=True)

            # Step 3: Process sequences with LSTM
            packed_lstm_out, _ = self.lstm(packed_sequences)

            # Step 4: Unpack the sequences
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

            # Step 5: Restore the original order
            _, unperm_idx = perm_idx.sort(0)
            lstm_out = lstm_out[unperm_idx]

            for i, length in enumerate(filtered_lengths):
                lstm_out_placeholder[non_zero_mask.nonzero(as_tuple=True)[0][i], length.int() - 1, :] = lstm_out[i, length.int() - 1, :]

        # Process spatial and property features
        spatial_out = self.spatial_net(spatial_features)
        print(f'printing spatial output {spatial_out}')
        property_out = self.property_net(property_features)

        # Calculate attention weights
        spatial_expanded = spatial_out.unsqueeze(1).repeat(1, lstm_out.size(1), 1)
        # print(len(sequences))
        # print(len(lstm_out))
        # print(len(spatial_out))
        # print(len(spatial_expanded))
        attention_input = torch.cat([lstm_out_placeholder, spatial_expanded], dim=2)
        attention_weights = torch.softmax(self.attention(attention_input), dim=1)
        # Apply attention
        context_vector = torch.sum(lstm_out_placeholder * attention_weights, dim=1)

        # Combine all features
        combined = torch.cat([context_vector, spatial_out, property_out], dim=1)
        combined = torch.cat([spatial_out, property_out], dim=1)
        # Make prediction
        prediction = self.predictor(combined)
        print(prediction)

        return prediction, attention_weights
