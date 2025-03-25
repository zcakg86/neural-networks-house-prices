#%%
import pandas as pd
import numpy as np
import torch
import joblib
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
        self.scalers = {}
        self.indices = []
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
        self.community_length = len(self.community_vocab)
        # Convert community IDs to indices using the vocabulary
        df['community_index'] = df['community'].map(self.community_vocab) # New column with indices
        #df['year_index']
        #df['week_index']
        self.length = df.shape[0]
        self.year_length = len(np.unique(df['year']))
        self.week_length = len(np.unique(df['week']))

        self.dataframe = df
    
    def _get_community_features(self):
        community_df = self.dataframe.groupby(['community_index', 'year']).agg({
            'sale_price': ['mean', 'median', 'std'],
            'sqft' : ['mean'],
            'beds' : 'median'
            }).to_dict('index')
        
        # Flatten the dictionary values
        for key, value in community_df.items():
            community_df[key] = [v for sublist in value.values() for v in (sublist if isinstance(sublist, list) else [sublist])]
        self.community_array = np.array([community_df[(c, y)] for c, y in zip(self.dataframe['community_index'], self.dataframe['year'])])
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
                                     torch.tensor(self.dataframe['log_price_scaled'].values.reshape(-1, 1), dtype=torch.float32))

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
