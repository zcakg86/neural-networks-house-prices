
#%%
from src.pricemodel.manager import *
pd.set_option('mode.chained_assignment', None)
from src.spatial.geotools import *
from src.spatial.community_detection import *
df = pd.read_csv('data/sales_2020_25.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df['price_per_sqft']=df['sale_price']/df['sqft']
df['sale_date']=pd.to_datetime(df['sale_date'])
df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
# And Zero values
df = df[df['sale_price'] > 0]
df = df[df['sqft'] > 0]
df = df[df['sale_nbr'] > 0]
#%%
df = df.h3.geo_to_h3(resolution = 8, lat_col = 'lat', lng_col = 'lng', 
                             set_index = False)
#%% 
cols_to_drop = df.columns[df.columns.str.contains('community|h3_index')]
df.drop(cols_to_drop, axis = 1, inplace = True)

def run_community_analysis(df, location_var, method, resolution=None):
    """
    Complete execution example
    """
    print("Creating location network...")
    G, location_features, features_df, array = create_location_network(df, location_var)
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nDetecting communities...")
    communities = detect_communities(G, method, resolution)
    print(f"Found {len(community_groups)} communities")

    return location_features, communities
locations, communities = run_community_analysis(df, location_var = 'h3_08', method = 'gn')


#location_features, G, features_df, array, communities, community_stats, \
#    community_groups = run_community_analysis(df, location_var = 'h3_08', method = 'gn', resolution = 0.3)
#%%
c_df = communities_df(community_groups)
df = df.merge(c_df, left_on = 'h3_08', right_on = 'h3_index')
h3_map(df, color ='community')
#%%
#print("\nCommunity Statistics:")
#print(stats)
#df = df.sample(n=100, random_state = 92)
#print(df.columns)
# area, city, subdivision, zoning, h3_10
# for i in ['area', 'city', 'subdivision', 'zoning', 'h3_10']:
    # print(i,'n unique : ',df[i].nunique(),'\n',df[i].unique())

#df = create_location_network(df, spatial_var)
# manager, df_with_predictions = train_and_save_model(df,sequence_length=10, epochs = 5)
#manager = train_and_save_model(df,sequence_length=3, epochs = 1)
# Create the scatter plot


# %%
