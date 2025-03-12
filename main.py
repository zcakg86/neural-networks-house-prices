#%%
from src.pricemodel.manager import *
df = pd.read_csv('data/sales_2020_25.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]

df = df.sample(n=100, random_state = 92)
manager, df_with_predictions = train_and_save_model(df,sequence_length=10, epochs = 5)

# %%
