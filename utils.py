from pyproj import Proj, transform
from shapely.geometry import box
import geopandas as gpd
# import pandas as pd
# import numpy as np
from tqdm import tqdm
import os



####################  Creating extent  ####################

def create_extent_from_centroid(src_crs, x, y, grid_width, grid_height, spatial_resolution):
    """
    Create top, left, bottom, right points from center points.
    First converts CRS to "EPSG:3857" and calculate four corner points by actual ground distance in meters.
    And converts back to original CRS.
    
    Args:
        src_crs (str) : coordinate reference system. e.g. "EPSG:4326"
        x (pandas Series) : longitude coordinates
        y (pandas Series) : latitude coordinates
        grid_width (int) : image pixel size. e.g. 512
        grid_height (int) : image pixel size. e.g. 512
        spatial_resolution(float) : ground distance in meters. e.g. 0.6 m
    
    Returns:
        a tuple of multiple pandas Series: (top, left, bottom, right)
    """
    
    crs_epsg_3857 = "EPSG:3857"
    
    # for changing crs
    src_Proj, epsg_3857_Proj = Proj(init=src_crs), Proj(init=crs_epsg_3857)

    # change crs to "EPSG:3857"
    x, y = transform(src_Proj, epsg_3857_Proj, x, y)
    
    # calculate four corner points from centroids
    top = y + (grid_height / 2) * spatial_resolution
    bottom = y - (grid_height / 2) * spatial_resolution
    left = x - (grid_width / 2) * spatial_resolution
    right = x + (grid_width / 2) * spatial_resolution
    
    # change crs back to src_crs
    left, top = transform(epsg_3857_Proj, src_Proj, left, top)
    right, bottom = transform(epsg_3857_Proj, src_Proj, right, bottom)
    
    return top, left, bottom, right


def produce_geojson(df, crs, columns, save_dir):
    """
    Produce a geojson file with grid polygons using top, left, bottom, right points from a pandas DataFrame.
    
    Args:
        df (pandas DataFrame): DataFrame containing four corner grid points (top, left, bottom, right)
        crs (str): crs of the grid corner points. e.g. "EPSG:4326", "EPSG:3857"
        columns (list): columns from DataFrame that will be included in geojson file. e.g. ["image_id", "lat", "lon", "top", "left", "bottom", "right"]
        save_dir (str): file path where geojson file will be saved
        
    Returns:
        None
    """
    
    # create grid polygons
    x_min_list = list(df["left"])
    y_min_list = list(df["bottom"])
    x_max_list = list(df["right"])
    y_max_list = list(df["top"])

    grids_list = []

    for xmin, ymin, xmax, ymax in zip(x_min_list, y_min_list, x_max_list, y_max_list):
        grid = box(xmin, ymin, xmax, ymax)
        grids_list.append(grid)
        
    # create GeoDataFrame
    geo_df = gpd.GeoDataFrame(df[columns], 
                              geometry = grids_list,
                              crs = crs)
    
    # save geojson file
    geo_df.to_file(save_dir)

    
    
####################  Add model prediction probability scores to dataframe  ####################

def add_yolov5_conf_scores(original_df, predictions_folder, school_conf_col = "conf_yolov5", image_id_col = "image_id"):
    """
    Get probability scores for school from YOLOv5 model prediction text files and add them to DataFrame.
    Text files contains predictions in (class, x, y, w, h, conf) format.
    
    Args:
        original_df (pandas DataFrame): YOLOv5 probability scores will be added to the DataFrame
        predictions_folder (str): folder containing YOLOv5 model prediction text files
        school_conf_col (str): column name for YOLOv5 probability scores in DataFrame
        image_id_col (str): column name for image ids in DataFrame
        
    Returns:
        DataFrame with YOLOv5 model probability scores added in a separate column
    """
    
    df = original_df.copy()

    # parse YOLOv5 prediction text files
    pred_text_list = os.listdir(predictions_folder)

    # loop through each prediction text file
    for filename in tqdm(pred_text_list):

        text_dir = os.path.join(predictions_folder, filename)

        with open(text_dir) as file:
            data = file.readlines()
            conf_list = []

            for line in data:
                line = line.strip()
                conf = float(line.split(' ')[-1])
                conf_list.append(conf)

            # multiple bbox could be detected, take the max conf score
            school_conf = max(conf_list)  

        # add current image prediction conf score to DataFrame
        df.loc[df[image_id_col] == int(filename.replace('.txt', '')), 
           school_conf_col] = school_conf

    return df


def add_efficientnet_conf_scores(original_df, predictions_folder, school_class_id = 1, school_conf_col = "conf_efficientnet", image_id_col = "image_id"):
    """
    Get probability scores for school from EfficientNet model prediction text files and add them to DataFrame.
    Text files contains predictions in (non_school_prob, school_prob) format.
    School class id is 1.
    
    Args:
        original_df (pandas DataFrame): EfficientNet probability scores will be added to this DataFrame
        predictions_folder (str): folder containing EfficientNet model prediction text files
        school_conf_col (str): column name for EfficientNet probability scores in DataFrame
        image_id_col (str): column name for image ids in DataFrame
        
    Returns:
        DataFrame with EfficientNet model probability scores added in a separate column
    """
    
    df = original_df.copy()
    
    pred_text_list = os.listdir(predictions_folder)

    # loop through each prediction text file
    for filename in tqdm(pred_text_list):

        text_dir = os.path.join(predictions_folder, filename)
        
        with open(text_dir) as file:
            data = file.readlines()[0]
            school_conf = data.split(' ')[school_class_id]
            school_conf = float(school_conf)

        df.loc[df[image_id_col] == int(filename.replace('.txt', '')), 
           school_conf_col] = school_conf
    
    return df