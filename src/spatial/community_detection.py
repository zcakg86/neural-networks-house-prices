import pandas as pd
import networkx as nx
import community  # python-louvain
import numpy as np
from scipy.stats import zscore

def create_location_network(df, location_var):
    """
    Creates network of locations based on similarity of sale_price patterns and characteristics
    
    Parameters:
    df: DataFrame with columns: 'location', 'sale_price', 'sale_date', and property characteristics
    """
    # Calculate location-level features
    location_features = {}
    
    for location in df[location_var].unique():
        loc_data = df[df[location_var] == location]
        
        price_metrics = {
            'mean_price': loc_data.groupby(pd.Grouper(key='sale_date', freq='M'))['sale_price'].mean(),
            'price_std': loc_data.groupby(pd.Grouper(key='sale_date', freq='M'))['sale_price'].std(),
            'price_per_sqm': loc_data.groupby(pd.Grouper(key='sale_date', freq='M'))(['sale_price']/['sqm']).mean(),
            'season_diff':loc_data.groupby(pd.Grouper(key='sale_date', freq='M'))['sale_price'].mean().diff(),
            'transaction_count': loc_data.groupby(pd.Grouper(key='sale_date', freq='M')).count(),
            'lng': loc_data['lng'].mean(),
            'lat': loc_data['lat'].mean()
        }

    location_features[location] = price_metrics
    
    # Create similarity network
    G = nx.Graph()
    
    # Add nodes
    for loc in location_features.keys():
        G.add_node(loc)
    
    # Add edges based on feature similarity
    for loc1 in location_features:
        feat1 = np.array(list(location_features[loc1].values()))
        
        for loc2 in location_features:
            if loc1 < loc2:  # Avoid duplicate edges
                feat2 = np.array(list(location_features[loc2].values()))
                
                # Calculate similarity (normalized Euclidean distance)
                similarity = 1 / (1 + np.linalg.norm(zscore(feat1) - zscore(feat2)))
                
                if similarity > 0.7:  # Threshold for edge creation
                    G.add_edge(loc1, loc2, weight=similarity)
    
    return G, location_features

def detect_communities(G):
    """
    Detects communities in the location network
    """
    # Use Louvain method for community detection
    communities = community.best_partition(G)
    
    # Group locations by community
    community_groups = {}
    for node, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        community_groups[community_id].append(node)
    
    return communities, community_groups

def analyze_communities(df, communities, location_features):
    """
    Creates summary statistics for each community
    """
    # Initialize community statistics
    community_stats = pd.DataFrame()
    
    for community_id, locations in communities.items():
        # Get data for all locations in community
        community_data = df[df['location'].isin(locations)]
        
        # Calculate community statistics
        stats = {
            'community_id': community_id,
            'num_locations': len(locations),
            'num_transactions': len(community_data),
            'avg_sale_price': community_data['sale_price'].mean(),
            'sale_price_std': community_data['sale_price'].std(),
            'avg_sale_price_per_sqm': (community_data['sale_price'] / community_data['size']).mean(),
            'locations': ', '.join(locations)
        }
        
        # Add time-based metrics if available
        if 'sale_date' in df.columns:
            monthly_sale_prices = community_data.groupby(
                pd.to_sale_datetime(community_data['sale_date']).dt.to_period('M')
            )['sale_price'].mean()
            
            stats.upsale_date({
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
def run_community_analysis(df):
    """
    Complete execution example
    """
    print("Creating location network...")
    G, location_features = create_location_network(df)
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nDetecting communities...")
    communities, community_groups = detect_communities(G)
    print(f"Found {len(community_groups)} communities")
    
    print("\nAnalyzing communities...")
    community_stats = analyze_communities(df, community_groups, location_features)
    
    return G, communities, community_stats

# Example usage with sample data:
"""
sample_data = pd.DataFrame({
    'location': ['A', 'A', 'B', 'B', 'C', 'C'],
    'sale_price': [100000, 120000, 200000, 220000, 150000, 160000],
    'size': [100, 120, 150, 160, 130, 140],
    'sale_date': ['2023-01-01', '2023-02-01', '2023-01-15', '2023-02-15', 
             '2023-01-01', '2023-02-01']
})

G, communities, stats = run_community_analysis(sample_data)
print("\nCommunity Statistics:")
print(stats)
"""