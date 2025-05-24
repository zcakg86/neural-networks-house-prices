#%%
#from src.pricemodel.embedding_model import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df = df.sample(n=1000, random_state = 92)

data = dataset()
data._prepare_data(df)
data._get_community_features()
# Scale and Create tensors
data._processor(mode = 'train')


embedding_dim=8
hidden_dim=8
property_dim=2


#%%
model = modelmanager(data)
model.train_model(embedding_dim, hidden_dim, property_dim, epochs = 1)

#%%
# %%
