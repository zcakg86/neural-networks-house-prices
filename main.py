
#%%
from src.pricemodel.manager import *
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt
from src.spatial.community_detection import *
from src.spatial.geotools import *
from src.spatial.community_detection import *

df = pd.read_csv('data/sales_2021_on_geo.csv')

df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df['price_per_sqft']=df['sale_price']/df['sqft']
df['sale_date']=pd.to_datetime(df['sale_date'])
df['h3_parent'] = vectorized_get_parent_h3(df['h3_10'], 7)


G, communities, stats = run_community_analysis(df, location_var = 'h3_parent')
print("\nCommunity Statistics:")
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
