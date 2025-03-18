#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

# Function to prepare data
class dataset:
    def __init__(self):
        self.length = None
        self.community_length = None
        self.year_length = None
        self.week_length = None
        self.scaler = StandardScaler()
        self.indices = []
        self.community_features = torch.empty(0)
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

        self.length = len(df)
        self.community_length = len(np.unique(df['community']))
        self.year_length = len(np.unique(df['year']))
        self.week_length = len(np.unique(df['week']))

        self.dataframe = df
    
    def _get_community_features(self):
        community_df = self.dataframe.groupby(['community', 'year']).agg({
            'sale_price': ['mean', 'median', 'std'],
            'sqft' : ['mean'],
            'beds' : 'median'
            }).to_dict('index')
        # Flatten the dictionary values
        for key, value in community_df.items():
            community_df[key] = [v for sublist in value.values() for v in (sublist if isinstance(sublist, list) else [sublist])]
        community_array = np.array([community_df[(c, y)] for c, y in zip(self.dataframe['community'], self.dataframe['year'])])
        self.community_features = torch.tensor(self.scaler.fit_transform(community_array), dtype=torch.float32)
        self.community_feature_dim = self.community_features.shape[1]

    def _processor(self):
        # Create tensor with each observation being contiguous, and scale fields.
        self.tensors = TensorDataset(torch.tensor(self.dataframe['community'].values, dtype=torch.int8),
                                     torch.tensor(self.dataframe['year'].values, dtype=torch.int8),
                                     torch.tensor(self.dataframe['week'].values, dtype=torch.int8),
                                     torch.tensor(self.scaler.fit_transform(self.dataframe[['sqft','price_per_sqft']].values), dtype=torch.float32),
                                     torch.tensor(self.scaler.fit_transform(self.dataframe['log_price'].values.reshape(-1, 1)), dtype=torch.float32))

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
        
        # Create sequences and features
        sequences, spatial_features, property_features, targets, self.indices, sequence_lengths = self._create_sequences(df)

        # Scale the features
        scaled_sequences = self._scale_non_zero(arr=sequences, scaler=self.scalers['sequences'])
        scaled_spatial = self.scalers['spatial'].fit_transform(spatial_features)
        scaled_property = self.scalers['property'].fit_transform(property_features)
        # Scale the target
        scaled_targets = self.scalers['prices'].fit_transform(targets.reshape(-1,1))

        return scaled_sequences, scaled_spatial, scaled_property, scaled_targets, df, sequence_lengths
    
    def _scale_non_zero(self, arr, scaler, dim=3):
    # Create a copy of the input array
        scaled_arr = arr.copy()
        # Reshape the array to (n*m, dim)
        reshaped = arr.reshape(-1, dim)
        # Create a mask for non-zero rows
        non_zero_mask = np.any(reshaped != 0, axis=1)
        # Extract non-zero rows
        non_zero_rows = reshaped[non_zero_mask]
        # Apply StandardScaler to non-zero rows
        scaled_non_zero = scaler.fit_transform(non_zero_rows)
        # Put scaled values back into the reshaped array
        reshaped[non_zero_mask] = scaled_non_zero
        # Reshape back to original dimensions
        scaled_arr = reshaped.reshape(arr.shape)
        
        return scaled_arr
    
    def _create_sequences(self, df):
        sequences = []
        spatial_features = []
        property_features = []
        targets = []
        sequence_indices = []
        sequence_lengths = []

        # Sequence creation looks for comparable properties
        for i in range(len(df)):
            current_property = df.iloc[i]
            comparables = self._find_comparable_properties(
                df, current_property, self.sequence_length
            )
            seq_features = []

            sequence_lengths.append(len(comparables))
            if sequence_lengths[i]>0:
                # Create sequence features

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
                    
            if sequence_lengths[i]<self.sequence_length:
                for _ in range(sequence_lengths[i]+1,self.sequence_length+1):
                    seq_features.append([0,0,0])

            spat_feat = [
                current_property['lat'],
                current_property['lng'],
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
                np.array(property_features), np.array(targets), 
                np.array(sequence_indices), np.array(sequence_lengths))

    def _find_comparable_properties(self, df, current_property, n_comparable=5,
                                    radius_km=1):
        """
        Find comparable properties based on location and characteristics, for sequencing
        """
        from sklearn.neighbors import BallTree

        # Create BallTree for spatial searching
        tree = BallTree(np.radians(df[['lat', 'lng']]), metric='haversine')

        # Find nearby properties. Search within radius and return observation index
        nearby_indices, distance = tree.query_radius(
            np.radians([[current_property['lat'], current_property['lng']]]),
            r=radius_km/6371.0,
            return_distance=True
        )

        # Find properties sold before current date
        nearby_df = df.iloc[nearby_indices[0]][['sale_date', 'sqft','log_price']].copy()
        nearby_df['distance'] = distance[0]
        
        # Calculate date difference
        # Calculate date difference
        nearby_df['days_diff'] = (current_property['sale_date'] - nearby_df['sale_date']).dt.days

        # Filter on date
        nearby_df = nearby_df[(nearby_df['days_diff'] > 0)
                              & (nearby_df['days_diff'] < 360)]

        # Calculate similarity scores based on sqft
        nearby_df['sqft_diff'] =   (nearby_df['sqft'] - current_property['sqft']
        ) / current_property['sqft']

        # Combine similarity scores (you can adjust weights)
        nearby_df['similarity_score'] = (
            0.4 * abs(nearby_df['sqft_diff']) +
            0.2 * nearby_df['days_diff'] / 365 +
            0.4 * nearby_df['distance']
        )
        # If more than minimum comparables, keep comparables and calculate differences for measure
        # similarity and return the n_comparable most similar
        if len(nearby_df) >= n_comparable:
            # Get most similar properties
            comparables = nearby_df.nsmallest(
                n_comparable, 'similarity_score'
            )
            
            nearby_df = comparables
        return nearby_df.iloc[:][['sale_date', 'sqft', 'log_price', 'distance']]
          
class maketensor(Dataset):
# To Create tensors from sequences/features.
    def __init__(self, sequences, spatial_features, property_features, targets, sequence_lengths):
        self.sequences = torch.FloatTensor(sequences)
        self.spatial_features = torch.FloatTensor(spatial_features)
        self.property_features = torch.FloatTensor(property_features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_lengths = torch.FloatTensor(sequence_lengths)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.spatial_features[idx],
                self.property_features[idx],
                self.targets[idx],
                self.sequence_lengths[idx])
# %%
