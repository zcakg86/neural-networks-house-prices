#%%
#from src.pricemodel.manager import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]

df = df.sample(n=100, random_state = 92)
#%%
#manager, df_with_predictions = train_and_save_model(df,sequence_length=10, epochs = 5)
#%%
# dataset will store information about the dataframe, once dataframe prepared
dataset = dataset()

df = dataset._prepare_data(df)


dataset.length

dataset._get_community_features(df)
#%%
dataset._processor(df)
#%%
dataset.property_features
#%%
dataset.community_features
#%% 
dataset.target

#%%
del dataset

#%%
dataset.week_length
#%%
dataset.length
#%%
dataset.year_length

# %%
