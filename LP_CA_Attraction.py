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
from pandas import DataFrame
#from tqdm import tqdm
import numpy as np

###############################################################################
###############################################################################

def raw_att_generation (gdf_data, uses_list, use_column, dist_list_regression,
                        att_type): 
     
    # list that will contain the tuples with pair uses
    evaluate_uses = []
    
    # iterate over the list of different uses
    for use in uses_list:
        
        # iterate again 
        for use2 in uses_list:
            
            # make each combination
            pair_uses_tuple = (use, use2)
            
            # add each pair tuple to the list
            evaluate_uses.append(pair_uses_tuple)
    
    d_total_area = dict.fromkeys(uses_list, 0)
    
    # iterate over the keys of the total areas dictionary
    for use in uses_list:
        
        # if the use of the parcel match the key, the area is being
        # added
        
        gdf_data_use = gdf_data[gdf_data[use_column] == use]
        d_total_area[use] = gdf_data_use["AREA"].sum()

    # add a total key to store the total area of the studied area
    d_total_area["total"] = gdf_data["AREA"].sum()
    
    # iterate ober the total area buffer to show basic characteristics of 
    # the studied area
    for use in d_total_area.keys():
        
        print(("Area " + str(use) + ": ").ljust(20) + str(d_total_area[use]) + " m2.")
    
    #--------------------------------------------------------------------------
    
    att_result = DataFrame(data = evaluate_uses, columns = ["origin_use", "evaluated_use"])
    proportion_uses_values = dict.fromkeys(evaluate_uses, 0)
    
    for dist in dist_list_regression:
        for use in uses_list:
            
            gdf_use = gdf_data[(gdf_data[use_column] == use)]

            for use2 in uses_list:
                
                if att_type =="vNI":
                    factor = 1
                elif att_type == "vF":
                    factor = (d_total_area[use2]/d_total_area["total"])
    
                value = gdf_use["BFF_" + str(dist) + "_PROPORTION_" + use2[0:2].upper()].mean()
                value = value/factor
                proportion_uses_values[use, use2] = value
        
        att_result[dist] = list(proportion_uses_values.values())

    return att_result   
     
#------------------------------------------------------------------------------

def calc_proportions (numpy_array1: np.array, numpy_array2: np.array):
    
    proportions = numpy_array1/numpy_array2
    
    return proportions

#------------------------------------------------------------------------------

def calc_att_value (a: np.array, b: np.array, c: np.array, d: np.array, dist: np.array) -> np.array:
    
    att_array = (a*(dist**3)) + (b*(dist**2)) + (c*dist) + d
    
    return att_array

#------------------------------------------------------------------------------

def calc_all_attraction (gdf, gdf_sj, df_coef, uses_list, simulation_column, dist):
    
    gdf =  gdf.sort_values("ID", ascending = True)
    
    for neighbour_use in uses_list:        
        for origin_use in uses_list:
            
            abrv_o = origin_use[0:2].upper()

            coefs = df_coef.loc[(df_coef["origin_use"] == origin_use) & (df_coef["evaluated_use"] == neighbour_use)].values.tolist()
            coefs = coefs[0][2:6]
            coef_names = ["a", "b", "c", "d"]

            for i in range (0, len(coef_names)):
                                
                gdf_sj.loc[(gdf_sj[simulation_column + "_1"] == neighbour_use), coef_names[i] + "_" + abrv_o] = coefs[i]

    for use in uses_list:
        abrv = use[0:2].upper()
        gdf_sj["ATR_" + abrv] = calc_att_value(gdf_sj["a_" + abrv].values,
                                               gdf_sj["b_" + abrv].values,
                                               gdf_sj["c_" + abrv].values,
                                               gdf_sj["d_" + abrv].values,
                                               gdf_sj["DIST"].values)
    for use in uses_list:
        
        abrv = use[0:2].upper()
        gdf_sj_use = gdf_sj.groupby(["ID_0"]).agg({"ATR_" + abrv: "sum"})
        gdf_sj_use = gdf_sj_use.sort_values("ID_0", ascending = True)
        
        gdf["ATR_" + abrv + "_UNST"] =  gdf_sj_use["ATR_" + abrv]
        gdf["ATR_" + abrv] = gdf["ATR_" + abrv + "_UNST"]

    return gdf

    






