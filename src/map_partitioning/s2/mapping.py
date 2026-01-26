import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_density_map(df: pd.DataFrame, title: str, x_label: str, y_label: str, 
                     gridsize: int = 60, figsize: tuple = (12, 8), 
                     x_lim: tuple = (-126, -66), y_lim: tuple = (24, 50)):
    """Generates a hexbin density plot of geographical points on a USA map."""
    print("Loading USA map for density plot...")
    usa_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    usa = gpd.read_file(usa_url)
    usa = usa[~usa['name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the USA map as background
    usa.plot(ax=ax, color='lightgray', edgecolor='black', zorder=1, alpha=0.7)

    # Hexbin for point density
    hb = ax.hexbin(df['longitude'], df['latitude'], gridsize=gridsize, 
                   cmap='viridis', mincnt=1, alpha=0.8, zorder=2)
    plt.colorbar(hb, ax=ax, label='Number of Images')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\nDensity Map Completed")