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
data.community_vocab
#%%
data.year_vocab
#%%
model = modelmanager(data,embedding_dim, hidden_dim, property_dim)
model.train_model(epochs = 5)

model.add_predictions_to_data()
#%%
data2 = data._processor(mode= 'test')
# %%
for i in data.tensors[:][0]:  
    print(data.tensors[i][0])

# %%
for i in model.dataset.tensors[:][0]:  
    print(model.dataset.tensors[i][0])
# %%
