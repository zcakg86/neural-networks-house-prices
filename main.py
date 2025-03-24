#%%
from src.pricemodel.manager import *
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]

df = df.sample(n=100, random_state = 92)

dataset = dataset()
dataset._prepare_data(df)

dataset._get_community_features()
# Scale and Create tensors
dataset._processor()

print(dataset.community_length)

# %%
model = modelmanager(dataset)

#%%
model.train_model(embedding_dim=8, hidden_dim=8, property_dim =8)
# %%
dataset.length
# %%
del model
# %%
torch.mps.is_available()
# %%
