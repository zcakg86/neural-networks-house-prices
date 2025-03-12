
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
#%%
def run_community_analysis(df, location_var, method, resolution=1):
    """
    Complete execution example
    """
    print("Creating location network...")
    G, location_features, features_df, array = create_location_network(df, location_var)
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nDetecting communities...")
    communities, summary = detect_communities(G, method, resolution)

    #print(f"Found {len(community_groups)} communities")

    return location_features, array, summary
# #%%
# locations, communities, community_summary = run_community_analysis(df, location_var = 'h3_08', method = 'gn')
# df['community'] = df['h3_08'].map(communities)
# df.to_csv(f'data/results/sales_202025.csv',
#                         index=False)
#%%
locations_l, communities_l, summary_l = run_community_analysis(df, location_var = 'h3_08', method = 'l', resolution = 1.5)
#%%
df['community'] = df['h3_08'].map(communities_l)
df.to_csv(f'data/sales_202025.csv',
                        index=False)
#%%
stats = analyze_communities(df, summary_l, location_var = 'h3_08')
#%%

df2 = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
h3_map(df2, color ='community')
# #%%
# h3_map(df, color ='sale_price',group_by='community')

#%%
h3_map(df2, color ='sale_price',group_by='community')
#%%
df2['community'].unique()
#%%
print(stats)
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
