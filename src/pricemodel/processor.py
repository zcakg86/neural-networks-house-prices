import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.pricemodel.models import *


# Class for working with data.
class data_processor:
    # Declare some objects within class on initialisation of object
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.scalers = {
            'sequences': StandardScaler(),
            'spatial': StandardScaler(),
            'property': StandardScaler(),
            'prices':StandardScaler()}
        self.indices = []
    # Function to prepare data
    def prepare_data(self, df):
        """
        Expected columns: ['sale_date', 'sale_price', 'lat', 'lng', 'sqft', 'sale_nbr','sqft_lot']
        """
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
        df['year'] = df['sale_date'].dt.year
        df['log_price']= np.log(df['sale_price'])

        # Calculate local market features
        df['local_avg_sqft'] = self._calculate_local_averages(df, 'sqft')

        # Create sequences and features
        sequences, spatial_features, property_features, targets, self.indices = self._create_sequences(df)
        # Scale the features
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        scaled_sequences = self.scalers['sequences'].fit_transform(sequences_reshaped)
        scaled_sequences = scaled_sequences.reshape(sequences.shape)
        scaled_spatial = self.scalers['spatial'].fit_transform(spatial_features)
        scaled_property = self.scalers['property'].fit_transform(property_features)
        # Scale the target
        scaled_targets = self.scalers['prices'].fit_transform(targets.reshape(-1,1))
        return scaled_sequences, scaled_spatial, scaled_property, scaled_targets, df


    def _calculate_local_averages(self, df, column, radius_km=2):
        """Calculate local averages within a radius"""
        # Should be for various time points
        from sklearn.neighbors import BallTree

        # Create BallTree for efficient nearest neighbor search
        tree = BallTree(np.radians(df[['lat', 'lng']]), metric='haversine')

        # Find neighbors within radius
        indices = tree.query_radius(np.radians(df[['lat', 'lng']]),
                                    r=radius_km / 6371.0)  # Convert km to radians

        # Calculate local averages
        local_avgs = []
        for idx_list in indices:
            local_avgs.append(df[column].iloc[idx_list].mean())

        return local_avgs

    def _create_sequences(self, df):
        sequences = []
        spatial_features = []
        property_features = []
        targets = []
        sequence_indices = []

        # Sequence creation looks for comparable properties
        for i in range(len(df)):
            current_property = df.iloc[i]
            comparables = self._find_comparable_properties(
                df, current_property, self.sequence_length
            )
            if comparables is not None:
                # Create sequence features
                seq_features = []
                for _, comp in comparables.iterrows():
                    # Price per sqft and other relative metrics
                    relative_size = comp['sqft'] / current_property['sqft']
                    days_before = (
                        current_property['sale_date'] - comp['sale_date']
                    ).days
                    # append price, size and date difference from comparabpes to use as sequential feature.
                    seq_features.append([
                        comp['log_price'],
                        relative_size,
                        days_before
                    ])

                # Spatial features (static)
                # also use comparables to create local spatial stats
                spat_feat = [
                    current_property['lat'],
                    current_property['lng'],
                    comparables['log_price'].median(),
                    comparables['sqft'].median()
                ]

                # Property features (static)
                prop_feat = [
                    current_property['sqft'],
                    current_property['sale_nbr'],
                    current_property['sqft_lot']
                ]

                sequences.append(seq_features)
                spatial_features.append(spat_feat)
                property_features.append(prop_feat)
                targets.append(current_property['log_price'])
                sequence_indices.append(i) 

        return (np.array(sequences), np.array(spatial_features),
                np.array(property_features), np.array(targets), sequence_indices)

    def _find_comparable_properties(self, df, current_property, n_comparable=5,
                                    radius_km=2):
          """
          Find comparable properties based on location and characteristics, for sequencing
          """
          from sklearn.neighbors import BallTree

          # Create BallTree for spatial searching
          tree = BallTree(np.radians(df[['lat', 'lng']]), metric='haversine')

          # Find nearby properties. Search within radius and return observation index
          nearby_indices = tree.query_radius(
              np.radians([[current_property['lat'], current_property['lng']]]),
              r=radius_km/6371.0
          )[0]

          # Find properties sold before current date
          nearby_df = df.iloc[nearby_indices].copy()
          prior_sales = nearby_df[
              # comparable properties must have been sold before and in the last year.
              (nearby_df['sale_date'] < current_property['sale_date'])
              & ((nearby_df['sale_date'] - current_property['sale_date']).dt.days < 360)
          ]
          # If more than minimum comparables, keep comparables and calculate differences for measure
          # similarity and return the n_comparable most similar
          if len(prior_sales) >= n_comparable:
              # Calculate similarity scores based on sqft
              prior_sales['sqft_diff'] = abs(
                  prior_sales['sqft'] - current_property['sqft']
              ) / current_property['sqft']

              # Sort by similarity and recency
              prior_sales['days_diff'] = (
                  current_property['sale_date'] - prior_sales['sale_date']
              ).dt.days

              # Combine similarity scores (you can adjust weights)
              prior_sales['similarity_score'] = (
                  0.7 * prior_sales['sqft_diff'] +
                  0.3 * prior_sales['days_diff'] / 365
              )

              # Get most similar properties
              comparables = prior_sales.nsmallest(
                  n_comparable, 'similarity_score'
              )

              return comparables

          return None
          
class maketensor(Dataset):
# To Create tensors from sequences/features.
    def __init__(self, sequences, spatial_features, property_features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.spatial_features = torch.FloatTensor(spatial_features)
        self.property_features = torch.FloatTensor(property_features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.spatial_features[idx],
                self.property_features[idx],
                self.targets[idx])