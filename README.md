# Deep learning for housing market analysis

Repository for training neural networks to observe markets and provide price prediction in Seattle and King County.\
Analysis and model performance to be presented with Panel.

## Aims
* Implement location-specific embedding through spatial features and community detection, to recuce spatial bias in price estimation across King County.
* Provide accurate estimates throughout time.
* Deliver dashboard with panel that can be used to present market analysis found through the model weights.

## Methods
* H3 Index for spatial representation
* Girvan-Newman community detection to allocate each H3 index to a community.
* Community embedding layer
* Temporal embedding layer based on YYYYMM and DD.
* Property Attribute data


## Data
Residential Sales data:\
Andy Krause https://www.kaggle.com/datasets/andykrause/kingcountysales/data Version 8: kingco_sales.csv\
Data produced from property assessment data made available by the King County Department of Assessments.\
See https://github.com/andykrause/kingCoData
