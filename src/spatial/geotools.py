# Using seaborn color palettes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h3
import h3pandas
import contextily as ctx

def vectorized_get_parent_h3(series, target_res):
    return series.map(lambda x: h3.cell_to_parent(x, target_res))

def h3_map(df, color):
    "Draw chloropleth based on 'color'"
    if color is None:
        color = 'h3_parent'
    
    if df[color].dtype == 'object': # for continuous variables
        colors = sns.color_palette("Paired",n_colors=len(df[color].unique()))
        color_map = dict(zip(df[color].unique(), colors))
    else: # if continuous
        color_map = sns.color_palette("bright")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect(1 / np.cos(np.radians(df['lat'].mean())))
    sns.scatterplot(data=df, x='lng', y='lat', hue=color, s=5, palette=color_map,
                     legend = False)
        
    # Convert lat/lon to Web Mercator projection
    ax.set_xlim(df['lng'].min(), df['lng'].max())
    ax.set_ylim(df['lat'].min(), df['lat'].max())

    # Add OSM background
    ctx.add_basemap(
        ax,
        crs='EPSG:4326',  # lat/lon projection
        source=ctx.providers.OpenStreetMap.Mapnik
    )
    
    plt.show()
    return ax