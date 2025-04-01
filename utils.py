import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import seaborn as sns


def load_labels(avalanches_path: Path, non_avalanches_path: Path):
    """Returns a dataframe of labeled locations. 

    Args:
        avalanches_path (Path): shp file with locations of avalanches.
        non_avalanches_path (Path): shp file with locations of non-avalanches.

    Returns:
        dataframe: The 'geometry' column specifies the location, and the boolean 'avalanche' column says if there was an avalanche at that location.
    """
    # Load coordinates of avalanches and non-avalanches
    avalanches_df = gpd.read_file(avalanches_path)
    non_avalanches_df = gpd.read_file(non_avalanches_path)
    
    # Make sure they use the same Coordinate Reference System (CRS)
    assert avalanches_df.crs.to_epsg() == non_avalanches_df.crs.to_epsg(), 'CRS do not match!'
    
    # Add 'avalanche' column, drop id, and drop rows w/ missing values
    # (non_avalanches_df has a missing value in the geometry column)
    avalanches_df = avalanches_df.drop(columns=['id']).dropna()
    avalanches_df['avalanche'] = True
    non_avalanches_df = non_avalanches_df.drop(columns=['id']).dropna()
    non_avalanches_df['avalanche'] = False

    # Concatenate rows to create single dataframe w/ all points of interest
    df = pd.concat([avalanches_df, non_avalanches_df], ignore_index=True)

    return df



class AvalancheFeatures:
    """Dataset of locations and corresponding features for avalanche risk assessment.

    Attributes:
        df: Dataframe with the features.
        file_to_col_name: dict with a mapping of tif filenames into dataframe columns.

    Methods:
        get_processed_X: returns dataframe with preprocessed features.
    """

    file_to_col_name = {
        'Aspect': 'aspect',
        'Elevation (m)': 'elevation',
        'Land_use': 'land_use',
        'Normalized Difference Snow Index': 'snow_index',
        'Plan curvature': 'plan_curvature',
        'Profile curvature': 'profile_curvature',
        'Terrain slope (degrees)': 'slope',
        'Winter precipitation (mm)': 'precipitation',
        'Winter_air_temperature': 'temperature'
    }

    def __init__(
            self, tif_paths: list[Path], coords: gpd.GeoSeries,
            verbose: bool = False
    ):
        """Creates a dataframe of avalanche predictors.

        The locations are specified by the 'geometry' series (the coords argument), which contains a list of (x, y) points. Each row in the dataframe corresponds to a single location, and each column holds the value of a single feature at that location. 

        Args:
            tif_paths (list[Path]): list of paths to tif files with features (elevation, land use, temperature etc) for the Shara mountain region.

            coords (geopandas.GeoSeries): the geometry series with the locations ((x, y) points) of locations of interest.

            verbose (bool): If true, reports the progress of loading tif files.
        """

        verbose_print = print if verbose else lambda *a, **k: None

        df = coords.copy().to_frame()

        # Need list of (x, y) for sampling from raster data
        coord_list = [
            (x, y) for x, y in zip(df['geometry'].x, df['geometry'].y)
        ]
        verbose_print(f'Received a total of {len(coord_list)} locations to sample')

        for path in tif_paths:
            verbose_print(f'Loading data from {path.name}')

            # Load tif data
            with rasterio.open(path) as src:

                assert df.crs.to_epsg() == src.crs.to_epsg(), 'CRS do not match!'

                # Get column name from filename
                col = self.file_to_col_name[path.stem]

                # Must check if sampled points are masked;
                # if they are, insert nan instead of default fill value
                masked_vals = [
                    x for x in src.sample(coord_list, masked=True)
                ]
                vals = [
                    mv.data[0] if not mv.mask else np.nan for mv in masked_vals
                ]
                # Add new column to DataFrame
                df[col] = vals

                verbose_print(f'Added {np.count_nonzero(~np.isnan(vals))} non-NaN values')

        # Cast land use as categorical
        df['land_use'] = df['land_use'].map({
            1: 'water',
            2: 'forest', 
            3: 'agricultural',
            4: 'settlement', 
            5: 'bare',
            6: 'snow',
            7: 'pasture'
        }).astype('category')

        # Cast aspect as categorical
        df['aspect'] = df['aspect'].map({
            1: 'north',
            2: 'north-east',
            3: 'east',
            4: 'south-east',
            5: 'south',
            6: 'south-west',
            7: 'west',
            8: 'north-west',
            9: 'flat'
        }).astype('category')

        # Store for future use
        self.df = df
    
    def get_ohe_land_use(self):
        """Returns a dataframe with 1-hot encoded land use values.
        """
        enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
        return enc.fit_transform(self.df['land_use'].to_frame())

    def get_sin_cos_enc_apect(self):
        """Returns a dataframe with sine-cosine encoded aspect values.
        """
        theta = self.df['aspect'].map({
            'north': 90, 'north-east': 45, 'east': 0, 'south-east': -45, 'south': -90, 'south-west': -135, 'west': -180, 'norht-west': 135
        })
        theta = np.deg2rad(theta)
        aspect_enc = pd.DataFrame(
            0.0, index=self.df.index, columns=['aspect_north', 'aspect_east']
        )
        aspect_enc.loc[:, 'aspect_north'] = np.cos(theta)
        aspect_enc.loc[:, 'aspect_east'] = np.sin(theta)
        # Since 9 (flat) wasn't mapped, it will show up as NaN.
        # Replacing these rows with [0, 0].
        aspect_enc = aspect_enc.fillna(0)
        return aspect_enc

    def get_X(self):
        """Returns design matrix X for passing to ML models.

        The precipitation and temperature features were obtained as linear mappings of elevation, so only the elevation feature is retained.

        The land use feature is a categorical variable and is one-hot encoded. The aspect feature represents the cardinal direction and is sine-cosine encoded.

        Returns:
            X (DataFrame): design matrix with feature values
        """

        # All variables except land_use, aspect, precipitation and temperature are copied directly
        drop_cols = [
            'land_use', 'aspect', 'precipitation', 'temperature', 'avalanche'
        ]
        cols = [c for c in self.df.columns if c not in drop_cols]
        X = self.df[cols]
        # feature_names = cols

        # Next 4 columns represent one-hot encoded land use
        land_use = self.get_ohe_land_use()
        X = pd.concat([X, land_use], axis=1)

        # Final two columns are sine-cosine encoded aspects
        aspect = self.get_sin_cos_enc_apect()
        X = pd.concat([X, aspect], axis=1)

        # Convert the geometry column into x and y coordinates
        points = X.pop('geometry')
        X.insert(0, 'x_coord', [p.x for p in points])
        X.insert(1, 'y_coord', [p.y for p in points])

        return X


def plot_relative_importances(
        importances, feature_names, title=None, save_path=None, fig_size=None
):
    """Create a plot showing relative importances of features.

    Args:
        importances (array-like): List of floats representing importances.
        feature_names (array-like): List of feature names.
        title (str): If provided, sets the figure title. Defaults to None.
        save_path (str): If provided, the figure is saved to the specified location.
        fig_size (float, float): If provided, controls the fig_size.
    """
    plt.figure(figsize=fig_size)
    sorting_idx = np.argsort(importances)
    plt.barh(range(len(importances)), importances[sorting_idx], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in sorting_idx])
    plt.xticks([])
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.xlabel('Relative importance')
    plt.show()


def load_fahp_scores(tif_path: Path, coords: gpd.GeoSeries):
    # Load scores of the FAHP model from a tiff file    

    # Need list of (x, y) for sampling from raster data
    coord_list = [
        (x, y) for x, y in zip(coords.x, coords.y)
    ]

    # Load tif data
    with rasterio.open(tif_path) as src:
        assert coords.crs.to_epsg() == src.crs.to_epsg(), 'CRS do not match!'

        # Must check if sampled points are masked;
        # if they are, insert nan instead of default fill value
        masked_vals = [
            x for x in src.sample(coord_list, masked=True)
        ]
        scores = [
            mv.data[0] if not mv.mask else np.nan for mv in masked_vals
        ]

    df = coords.copy().to_frame()
    df['fahp_score'] = scores

    return df


def test_fahp_loading(
        avalanches_path, non_avalanches_path, fahp_scores_path
):
    labels = load_labels(
        avalanches_path=avalanches_path,
        non_avalanches_path=non_avalanches_path
    )

    fahp_scores = load_fahp_scores(
        tif_path=fahp_scores_path,
        coords=labels['geometry']
    )
    df =  fahp_scores.sjoin(labels, how="inner", predicate='intersects')
    print(df.describe())
    sns.histplot(data=df, x='fahp_score', hue='avalanche')
    plt.show()


def test_with_labeled_locations(
    avalanches_path, non_avalanches_path, tif_paths        
):
    labels = load_labels(
        avalanches_path=avalanches_path,
        non_avalanches_path=non_avalanches_path
    )

    features = AvalancheFeatures(
        tif_paths=tif_paths,
        coords=labels['geometry']
    )

    print(features.df.head())
    print(features.df.describe())

    # Drop data points with NaN in any column
    X = features.get_X()
    nan_idx = X[X.isna().any(axis=1)].index
    X = X.drop(index=nan_idx)
    labels = labels.drop(index=nan_idx)
    y = labels['avalanche'].values.astype(int)

    print(f'X.shape = {X.shape}, y.shape = {y.shape}')
    print(f'Feature names: {X.columns}')


def test_with_land_use_locations(tif_paths, save_path=None):
    # Find Land_use.tif path
    land_use_path = [p for p in tif_paths if p.stem == 'Land_use']

    with rasterio.open(land_use_path[0]) as src:
        rows, cols = np.nonzero(src.dataset_mask())
        x, y = src.xy(rows, cols)
        crs = src.crs
    coords = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)

    features = AvalancheFeatures(
        tif_paths=tif_paths, 
        coords=coords['geometry'].sample(frac=1),
        verbose=True
    )
    
    # print(features.df.head())
    # print(features.df.describe())

    # Drop data points with NaN in any column
    X = features.get_X()
    nan_idx = X[X.isna().any(axis=1)].index
    X = X.drop(index=nan_idx)

    # print(f'X.shape = {X.shape}')
    # print(f'Feature names: {X.columns}')

    if save_path is not None:
        X = X.astype('float32')
        with open(save_path, 'wb') as f:
            pickle.dump(X, f)


if __name__ == '__main__':

    do_test_with_labeled_locations = False
    do_test_with_land_use_locations = False
    do_test_fahp_loading = True

    # Define path to the data files
    data_path = Path('./data')
    tif_files = list(data_path.glob('*.tif'))

    # Precipitation and temperature were obtained as linear mappings of elevation, and are therefore useless.
    tif_files = [
        p for p in tif_files if p.stem not in 
        {'Winter_air_temperature', 'Winter precipitation (mm)'}
    ]

    if do_test_with_labeled_locations:
        test_with_labeled_locations(
            avalanches_path = data_path / 'Avalanches.shp',
            non_avalanches_path = data_path / 'Non avalanches.shp',
            tif_paths = [file for file in tif_files]
        )

    if do_test_with_land_use_locations:
        test_with_land_use_locations(
            tif_paths = [data_path / file for file in tif_files],
            save_path = data_path / 'features_all_coords.pkl'
        )

    if do_test_fahp_loading:
        test_fahp_loading(
            avalanches_path = data_path / 'Avalanches.shp',
            non_avalanches_path = data_path / 'Non avalanches.shp',
            fahp_scores_path = Path('./results/FAHP_sintezna.tif')
        )

    pass