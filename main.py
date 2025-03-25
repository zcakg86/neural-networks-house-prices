#%%
from src.pricemodel.manager import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]

df = df.sample(n=1000, random_state = 92)

dataset = dataset()
dataset._prepare_data(df)
dataset._get_community_features()
# Scale and Create tensors
dataset._processor()

print(dataset.community_length)
model = modelmanager(dataset)

#%%
from src.spatial.geotools import *
h3_map(dataset.dataframe, color='price_per_sqft')
#%%
#print(dataset.community_length)
#print(dataset.tensors.tensors)
print(df.columns)#

#%%
model.train_model(embedding_dim=8, hidden_dim=8, property_dim=8)
# %%
# model1 = embeddingmodel(dataset, dataset.community_length,
#                         8, 8, 8)
# %%
del model
# %%
torch.mps.is_available()
# %%
