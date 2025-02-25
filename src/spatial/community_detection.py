#%%
import pandas as pd
import networkx as nx
import community  # python-louvain
import numpy as np
from scipy.stats import zscore

def create_location_network(df, location_var=None):
    """
    Creates network of locations based on similarity of sale_price patterns and characteristics
    
    Parameters:
    df: DataFrame with columns: 'location', 'sale_price', 'sale_date', and property characteristics
    """
    # Calculate location-level features
    location_features = {}
    print(f'Locations: {df[location_var].unique().shape}')

    for location in df[location_var].unique():
        loc_data = df[df[location_var] == location]
        metrics = {
            'price_per_sqft': loc_data.groupby(pd.Grouper(key='sale_date', freq='QE'))['price_per_sqft'].mean(),
            'sqft': loc_data.groupby(pd.Grouper(key='sale_date', freq='QE'))['sqft'].mean(),
            'sqft_std': loc_data.groupby(pd.Grouper(key='sale_date', freq='QE'))['sqft'].std(),
            'beds': loc_data.groupby(pd.Grouper(key='sale_date', freq='QE'))['sale_nbr'].median(),
            'lng': loc_data['lng'].mean(),
            'lat': loc_data['lat'].mean()
        }
        location_features[location] = metrics
    print(list(location_features.values())[0:5])
    print(list(location_features.keys())[0:5])

    # Create graph
    G = nx.Graph()
    # Add nodes
    for loc in location_features.keys():
        G.add_node(loc)

    # Create array with location codes and features
    features_array = []
    locations = list(location_features.keys())

    for loc in locations:
        # Extract values from the nested dictionary/series
        loc_features = [
            location_features[loc]['price_per_sqft'].mean(),
            location_features[loc]['sqft'].mean()#,
            # location_features[loc]['sqft_std'].mean(),
            # location_features[loc]['beds'].mean(),
            # location_features[loc]['lat'],
            # location_features[loc]['lng']
        ]
        features_array.append(loc_features)
    
    # Convert to numpy array and standardize
    features_array = np.array(features_array)
    print(features_array[0:5])
    features_standardized = zscore(features_array, axis=0, nan_policy='omit')
    print(features_standardized[0:5])
    # Create DataFrame with location codes and standardized features
    features_df = pd.DataFrame(
        features_standardized, 
        index=locations,  # This keeps location codes as index
        columns=['price_per_sqft' ,'sqft' 
                 #,'sqft_std', 'beds', 'lat', 'lng'
                 #,season_diff'
        ]
    )

    # Create edges using standardized features
    for loc1 in features_df.index:
        for loc2 in features_df.index[features_df.index > loc1]:  # More efficient way to avoid duplicates
            similarity = 1 / (1 + np.linalg.norm(features_df.loc[loc1] - features_df.loc[loc2]))
            if similarity >= 0.5:
                G.add_edge(loc1, loc2, weight=similarity)
            if loc1 in features_df.index[0:5] & loc2 in features_df.index[0:5]:
                print(f'{loc1} and {loc2} similarity: {similarity}')
    
    return G, location_features, features_df, features_array


def detect_communities(G, method = 'gn', res=1):
    """
    Detects communities in the location network
    """
    # Use Louvain method for community detection
    print(f'Resolution: {res}')
    if method == 'gn':
        communities = nx.community.girvan_newman(G,\
                        max(G.edges.items(),
                            key=lambda edge: edge[1]['weight'])[0])
    elif method == 'l':
        communities = nx.community.louvain_communities(G, resolution=res)
    
    # # Group locations by community
    # community_groups = {}
    # for node, community_id in communities:
    #     if community_id not in community_groups:
    #         community_groups[community_id] = []
    #     community_groups[community_id].append(node)
    
    return communities#, community_groups

def communities_df(input):
    '''Create datafame from dict'''
    df = pd.DataFrame([(k, i) for k, v in input.items() for i in v], 
                      columns=['community', 'h3_index'])
    return df
    

def analyze_communities(df, communities, location_var):
    """
    Creates summary statistics for each community
    """
    # Initialize community statistics
    community_stats = pd.DataFrame()
    
    for community_id, locations in communities.items():
        # Get data for all locations in community
        community_data = df[df[location_var].isin(locations)]
        
        # Calculate community statistics
        stats = {
            'community_id': community_id,
            'num_locations': len(locations),
            'num_transactions': len(community_data),
            'avg_sale_price': community_data['sale_price'].mean(),
            'sale_price_std': community_data['sale_price'].std(),
            'price_per_sqft': community_data['price_per_sqft'].mean(),
            'locations': ', '.join(locations)
        }
        
        # Add time-based metrics if available
        if 'sale_date' in df.columns:
            monthly_sale_prices = community_data.groupby(
                pd.to_datetime(community_data['sale_date']).dt.to_period('M')
            )['sale_price'].mean()
            
            stats.update({
                'sale_price_trend': monthly_sale_prices.pct_change().mean(),
                'sale_price_volatility': monthly_sale_prices.pct_change().std()
            })
        
        # Add to community statistics
        community_stats = pd.concat([
            community_stats,
            pd.DataFrame([stats])
        ])
    
    return community_stats.reset_index(drop=True)

# Example execution:
def run_community_analysis(df, location_var, method, resolution=None):
    """
    Complete execution example
    """
    print("Creating location network...")
    G, location_features, features_df, array = create_location_network(df, location_var)
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nDetecting communities...")
    communities, community_groups = detect_communities(G, method, resolution)
    print(f"Found {len(community_groups)} communities")
    
    print("\nAnalyzing communities...")
    community_stats = analyze_communities(df, community_groups, location_var)
    
    return location_features, G, features_df, array, communities, community_stats, community_groups

# # Example usage with sample data:
# sample_data = pd.DataFrame({
#     'location': ['A', 'A', 'B', 'B', 'C', 'C'],
#     'sale_price': [100000, 120000, 200000, 220000, 150000, 160000],
#     'sqft': [100, 120, 150, 160, 130, 140],
#     'sale_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-15', '2023-02-15','2023-01-01', '2023-02-01']),
#     'lat':[47.5,47.6,47.55,47.55,47.6,47.53],
#     'lng':[-122.23,-122.24,-122.24,-122.43,-122.42,-122.32]
# })
# sample_data['price_per_sqft']=sample_data['sale_price']/sample_data['sqft']

# G, communities, stats = run_community_analysis(sample_data, location_var = 'location')
# print("\nCommunity Statistics:")
# print(stats)

# %%
