# -*- coding: utf-8 -*-
"""
@Author: UAH - TRANSURBAN - Nikolai Shurupov - Ramón Molinero Parejo
@Date: Thu Dec 10 17:15:09 2020
@Version: 2.0
@Description: Disruptive vetorial cellullar automata simulation module.
"""

###############################################################################
###############################################################################

# global imports
from geopandas import GeoDataFrame, sjoin
import numpy as np

###############################################################################
###############################################################################

def group_by_neigh_IDs(gdf, use_column):
        
    keys, _, _, values, _, _, _ = gdf.sort_values('ID_0').values.T
    ukeys, index = np.unique(keys, True)
    
    areas_array = np.split(values, index[1:])

    gdf_result = GeoDataFrame({'ID_0':ukeys, "AREA_1":[list(a) for a in areas_array]})
    return gdf_result

#------------------------------------------------------------------------------

def calc_distance_XY (x_0: np.array, x_1: np.array,
                      y_0: np.array, y_1: np.array) -> np.array:
                                        
    distance_array = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
    
    return distance_array

#------------------------------------------------------------------------------

def neighbours_vectorized (gdf, dist, uses_list, use_column):
    
    gdf = gdf.sort_values("ID", ascending = True)
    
    gdf_ID = GeoDataFrame()
    gdf_ID["Index"] = gdf["ID"]
    gdf_ID = gdf_ID.set_index("Index")
    gdf_ID["ID"] = gdf_ID.index
    
    # generate a new GDF copying spatial relevant fields
    gdf_geom = gdf[{"ID", "AREA", use_column, "COOR_X", "COOR_Y", "geometry"}]
    
    # create a copy GDF of the geometry one
    gdf_geom_buffer = gdf_geom.copy(deep = True)
    
    # perform a buffer to each parcel of the copied GDF
    gdf_geom_buffer["geometry"] = gdf_geom.geometry.buffer(dist)
    
    # perform a spatial join, so that each ID of the first GDF of 
    # geometries will have multiple ID of the copied one
    spatial_join = sjoin(gdf_geom_buffer, gdf_geom, how="inner", op="intersects", 
                             lsuffix="0", rsuffix="1")
    
    sj_not_self_intersection = spatial_join[spatial_join["ID_0"] != spatial_join["ID_1"]]
    
    sj_not_self_intersection = sj_not_self_intersection[["ID_0", "ID_1",
                               "AREA_0", "AREA_1",use_column + "_0",
                               use_column + "_1", "geometry"]]
            
    sj_group_by_ID = group_by_neigh_IDs(sj_not_self_intersection, use_column)
    sj_group_by_ID = sj_group_by_ID.sort_values('ID_0')
   
    sj_total_area_use =  sj_not_self_intersection.groupby(["ID_0"]).agg({"AREA_1": "sum"})

    sj_area_use =  sj_not_self_intersection.groupby(["ID_0", use_column + "_1"]).agg({"AREA_1": "sum"})
    sj_area_use_unstacked = sj_area_use.unstack(level = use_column + "_1")
    sj_area_use_unstacked = sj_area_use_unstacked.sort_values("ID_0", ascending = True)
    
    for use in uses_list:
        gdf_ID["BFF_" + "_AREA_USE_" + use[0:2].upper()] =\
            sj_area_use_unstacked["AREA_1", use]
            
    gdf_ID["TOTAL_N_AREA"] = sj_total_area_use["AREA_1"]

    return gdf_ID

#------------------------------------------------------------------------------

def generate_spatial_join(gdf, simulation_column, dist):
    
    gdf = gdf.sort_values("ID", ascending = True)

    # generate a new GDF copying spatial relevant fields
    gdf_geom = gdf[{"ID", "AREA", simulation_column, "COOR_X", "COOR_Y", "geometry"}]

    # create a copy GDF of the geometry one
    gdf_geom_buffer = gdf_geom.copy(deep = True)

    # perform a buffer to each parcel of the copied GDF
    gdf_geom_buffer["geometry"] = gdf_geom.geometry.buffer(dist)

    # perform a spatial join, so that each ID of the first GDF of 
    # geometries will have multiple ID of the copied one
    spatial_join = sjoin(gdf_geom_buffer, gdf_geom, how="inner", op="intersects", 
                             lsuffix="0", rsuffix="1")

    sj_not_self_intersection = spatial_join[spatial_join["ID_0"] != spatial_join["ID_1"]]

    sj_not_self_intersection["DIST"] = calc_distance_XY(sj_not_self_intersection["COOR_X_0"].values,
                                                        sj_not_self_intersection["COOR_X_1"].values,
                                                        sj_not_self_intersection["COOR_Y_0"].values,
                                                        sj_not_self_intersection["COOR_Y_1"].values)
    return sj_not_self_intersection



            
    






