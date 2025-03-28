#%%
#from src.pricemodel.embedding_model import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df = df.sample(n=100, random_state = 92)

dataset = dataset()
dataset._prepare_data(df)
dataset._get_community_features()
# Scale and Create tensors
dataset._processor(mode = 'train')

#%%
embedding_dim=8
hidden_dim=8
property_dim=2

model = modelmanager(dataset)
model.train_model(embedding_dim, hidden_dim, property_dim)

trained_model = manager.model



#%%


dataset.community_array[:,0]
# %%
dataset.community_df
# %%
