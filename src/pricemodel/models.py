#%%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_forecasting.models.nn.rnn import LSTM

class embeddingmodel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, property_dim, 
                 community_length,community_feature_dim, week_length, year_length):
        # inherit from nn.Module
        super().__init__()
        # Layer dims
        self.property_dim = property_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Embedding dims
        self.community_embedding_length= community_length
        self.community_feature_dim = community_feature_dim
        self.week_length = week_length
        self.year_length = year_length

        # Embedding Layers
        self.community_embedding = nn.Embedding(int(self.community_embedding_length), embedding_dim)
        self.year_embedding = nn.Embedding(int(self.year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(self.week_length), embedding_dim)

        # Feature Processing Layers
        self.community_feature_layer = nn.Linear(self.community_feature_dim, hidden_dim)
        self.property_feature_layer = nn.Linear(property_dim, hidden_dim)

        # Combine embedding dimension and processed feature dimensions
        combined_embedding_dim = 3 * embedding_dim
        input_dim = combined_embedding_dim + 2 * hidden_dim # Two hidden_dim from processed features

        # Attention Layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True) 

        # Hidden and Output Layers
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)  # Input now includes attention output
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)  # Optional additional hidden layer
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

        # Combine embeddings and features
        combined_features = torch.cat([combined_embeddings, processed_community_features, processed_property_features], dim=-1)

        # Reshape for attention layer (requires 3D tensor: batch_size x seq_len x features)
        # In this case, seq_len = 1 because we're treating each property as a single sequence
        combined_features = combined_features.unsqueeze(1)

        # Attention Layer
        attention_output, _ = self.attention_layer(combined_features, combined_features, combined_features) # Self-attention
        attention_output = attention_output.squeeze(1) # Remove sequence dimension

        # Hidden Layers (after attention)
        hidden1 = self.relu(self.hidden_layer1(attention_output))
        hidden2 = self.relu(self.hidden_layer2(hidden1))

        # Output Layer
        output = self.output_layer(hidden2)
        return output


class initialmodel(nn.Module):
    def __init__(self, sequence_dim, spatial_dim, property_dim, embedding_dim, hidden_dim=64):
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

# %%
