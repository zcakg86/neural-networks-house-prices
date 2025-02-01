from src.pricemodel.manager import *
pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('data/sales_2021_on_geo.csv')
df = df[df['lat'].between(47.55,47.64) & df['lng'].between(-122.32,-122.27)].sample(n=100, random_state = 92)
manager, df_with_predictions = train_and_save_model(df,sequence_length=10, epochs = 50)
