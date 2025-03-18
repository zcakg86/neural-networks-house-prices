#%%
from src.pricemodel.manager import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]

df = df.sample(n=100, random_state = 92)
#%%
#manager, df_with_predictions = train_and_save_model(df,sequence_length=10, epochs = 5)
#%% dataset will store information about the dataframe, once dataframe prepared
dataset = dataset()
dataset._prepare_data(df)
#%% find aggregate features
dataset._get_community_features()
#%% Scale and Create tensors
dataset._processor()
#%%
print(dataset.community_length)
#%%
model = embeddingmodel()
#%%
model = embeddingmodel(dataset, embedding_dim = 8, hidden_dim = 8, property_dim = 6)
# %%
# %%
