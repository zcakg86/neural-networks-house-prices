# Load pre-trained model
from src.pricemodel.manager import *

loaded_predictor, loaded_processor = load_saved_model_with_config(path_prefix='outputs/models/property_model_20250130_014734')
print(loaded_processor.scalers['temporal'].var_[0])
print(loaded_processor.scalers['temporal'].mean_[0])
test_df = pd.read_csv('data/sales_2021_on_geo.csv').sample(n=500, random_state = 2)
manager = modelmanager(loaded_predictor, loaded_processor)
sequences, spatial_features, property_features, targets, df_transformed = loaded_processor.prepare_data(test_df)
manager.add_predictions_to_data(test_df, sequences, spatial_features, property_features)