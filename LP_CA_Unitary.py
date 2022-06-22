# -*- coding: utf-8 -*-
"""
@Author: UAH - TRANSURBAN - Nikolai Shurupov - Ramón Molinero Parejo
@Date: Thu Dec 10 17:15:09 2020
@Version: 1.1
@Description: Disruptive vetorial cellullar automata simulation module.
              Encharged of the generation of neighbours for each parcel based
              on different selected buffer distances.
"""

###############################################################################
###############################################################################

# global imports
from pandas import DataFrame, read_csv, concat
from os import chdir, makedirs, path
from geopandas import read_file, GeoDataFrame, sjoin
from time import strftime, localtime, time
#from tqdm import tqdm
from copy import deepcopy
import numpy as np
from numpy import array, var, nansum

# imports to make plots
from matplotlib import pyplot as plt

# imports to perform regressions
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as MSE

from random import uniform
from math import log

###############################################################################
###############################################################################


def writeLog(Msg, outFile):
    '''
    Write an input string on an input file. Has te objectivo fo tracing all the
    relevant processes that the model performs, storing it on a file as well as
    printing it to console.
    '''
    
    # print on console the string
    print (Msg)
    
    # wrinte the string on the output file after making a line break
    outFile.write("\n" + Msg)
   
#------------------------------------------------------------------------------

def f_poly_g3(x, a, b, c, d):
    return (a*(x**3)) +( b*(x**2)) + (c*x) + d

#------------------------------------------------------------------------------

def compute_r2(y_true, y_pred):
    '''
    Basic function to evaluate the R2 of a data set. List/array with values of
    observed Y and then ones generated with the regression function.
    '''
    sse = sum((y_true - y_pred)**2)
    tse = (len(y_true) - 1) * var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score

#------------------------------------------------------------------------------

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

def frc_definer (gdf, frc, uses_list, simulation_column):
    
    for use in uses_list:
        gdf.loc[gdf[simulation_column] != use, "FRC_"  + use[0:2].upper()] = frc
        
    for use in uses_list:
        gdf.loc[(gdf[simulation_column] == use) | (gdf[simulation_column] == 'vacant'), "FRC_"  + use[0:2].upper()] = 1
        
    return gdf

#------------------------------------------------------------------------------

def norm2(gdf, potential_use_list):
    '''
    Normalize the attraction values of an inputted DataFrame. It treats all 
    values of attraction as a whole, making an unique set of data using all the
    columns that fits in the potential use list.
    '''
    
    # define a list that will store all the values of the different columns 
    all_values_list = []
    
    # fill the list with all column of previously generated not normalized 
    # attraction values
    for use in potential_use_list:
        list_field = gdf["ATR_" + use[0:2].upper() + "_UNST"].tolist()
        all_values_list.extend(list_field)
    
    # compute the max, min and difference of the set of values
    max_value = max(all_values_list)
    min_value = min(all_values_list)
    difference = (max_value - min_value)
    
    # iterate for each use
    for use in potential_use_list:
        
        # get the list of attraction values (the not normalized one)
        gdf = gdf.sort_values("ATR_" + use[0:2].upper() + "_UNST", ascending = True)
        list_field = gdf["ATR_" + use[0:2].upper() + "_UNST"].tolist()
        
        # define a variable to store the normalized values
        list_norm = []
        
        # normalize the values of the list of values
        for x in list_field:
            y = (x - min_value) / difference
            list_norm.append(y)
        
        # update the normalized attraction columns with the new values
        gdf["ATR_" + use[0:2].upper()] = list_norm
        
        # sort the GDF using ID, as it previously was
        gdf = gdf.sort_values("ID", ascending = True)
        
    del all_values_list, list_field, list_norm, max_value, min_value,\
        difference, y, x

    return gdf

#------------------------------------------------------------------------------

def norm(gdf , field):
    '''
    Normalize a field(column) of an inputted DataFrame. Name of one field must
    be given as input
    ''' 
    
    # distance field will be inverted, for any other, sort the GDF using the
    # field from min to max
    if field in ["DIST", "SLOPE"]:
        gdf = gdf.sort_values(field, ascending = False)
    else:
        gdf = gdf.sort_values(field, ascending = True)
     
    # list with all the values of the field
    list_field = gdf[field].tolist()
    
    # define a variable to store the normalized values
    list_norm = []
    
    # compute the max, min and difference of the set of values
    max_value = max(list_field)
    min_value = min(list_field)
    difference = (max_value - min_value)
    
    # normalize the values of the list of values
    for x in list_field:
        y = (x - min_value) / difference
        list_norm.append(y)
    
    # make sure its sorted from min to max, in case DIST field was selected
    gdf = gdf.sort_values(field, ascending = True)    

    # update the values with the normalized values
    gdf[field] = list_norm
    
    # sort the GDF using ID, as it previously was
    gdf = gdf.sort_values("ID", ascending = True)
    
    del list_field, list_norm, max_value, min_value, difference, y, x
    
    return gdf

#------------------------------------------------------------------------------

def alpha_generator(x, alfa):
    return x + (-log(uniform(0, 1)))**alfa

#------------------------------------------------------------------------------

def dyn_bool_definer (gdf, use_column, evaluation_column):
    
    gdf["DYN_BOOL"] = 0
    gdf.loc[gdf[use_column] != gdf[evaluation_column], "DYN_BOOL"] = 1
    
    return gdf

#------------------------------------------------------------------------------

def calc_att_value (a: np.array, b: np.array, c: np.array, d: np.array, dist: np.array) -> np.array:
    
    att_array = (a*(dist**3)) + (b*(dist**2)) + (c*dist) + d
    
    return att_array

#------------------------------------------------------------------------------

def calc_y_pred (x: np.array, a, b, c, d) -> np.array:
    
    att_array = (a*(x**3)) + (b*(x**2)) + (c*x) + d
    
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

#------------------------------------------------------------------------------

def update_sj_use_changes(gdf_sj, simulation_column, use, list_IDs):

    gdf_sj.loc[gdf_sj["ID_1"].isin(list_IDs), simulation_column + "_1"] = use
    
    return gdf_sj
    
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

#------------------------------------------------------------------------------

def potentials (array_ACCESS: np.array, array_ATR: np.array,array_FRC: np.array,
                array_SUITABILITY: np.array, array_ZONING: np.array, alfa):
    
    array_ALFA = np.ones(len(array_ATR))
    array_ALFA = np.vectorize(alpha_generator)(array_ALFA, alfa)
    result_array = array_ACCESS * array_ATR * array_FRC * array_SUITABILITY * array_ZONING * array_ALFA
    #result_array = array_ATR * array_FRC * array_APTITUDE * array_ALFA
                            
    return result_array

#------------------------------------------------------------------------------

def get_order_pot (row, potential_use_list):
    
    uses_values = []
    pot_sorted_names = []
    
    for use in potential_use_list:
        
        abrv = use[0:2].upper()
        pot_value = row["POT_" + abrv]
        
        if pot_value > 0:
            uses_values.append(pot_value)        
        
    # the list with all the values is sorted, from max to min
    ls_pot_sorted = deepcopy(uses_values)
    ls_pot_sorted.sort(reverse = True)
        
    # position of the maximum is looked for and based on that new list with
    # the name in descending order of each use is stored to the row
    for item in ls_pot_sorted:
        
        use_position = uses_values.index(item)
        pot_sorted_names.append(potential_use_list[use_position])
        
    return pot_sorted_names

#------------------------------------------------------------------------------

def simulation(gdf, df_coef, initial_year, final_year, frc, alfa, cicle, outFile,
               use_column, evaluation_column, simulation_column, dist,
               demand_values, uses_list, gdf_sj):
    '''
    # Funcion that performs the simulation
    #--------------------------------------------------------------------------
    # Input parameters requirements:
    # - gdf: GeoDataFrame of the parcels
    # - df_coef: DataFrame with the regression coefficents to compute attraction
    #            values
    # - YYYY: Final year to simulate
    # - frc: Resistance to change factor (0.9, 0.8, 0.7 ...)
    # - alfa: Random factor (0.9, 0.8, 0.7 ...)
    # - cicle: How many year represent each iteration (1, 2, 5, 10 ...)
    # - outFile: File to write the Logs on
    #--------------------------------------------------------------------------
    # Output:
    #    GeoDataFrame with a columns of the simulated use for each parcel
    '''
    
    #--------------------------------------------------------------------------
    writeLog("Preparing the simulation...", outFile) 
    
    # list with the parcel uses that are able to change
    uses_to_change_list = deepcopy(uses_list)
    uses_to_change_list.append("vacant")
    
    for use in uses_list:

        abrv = use[0:2].upper()
        gdf["POT_" + abrv] = 0.0

    demand_co_tot = demand_values["DEMAND_CO"]
    demand_in_tot = demand_values["DEMAND_IN"]
    demand_si_tot = demand_values["DEMAND_SI"]
    demand_mu_tot = demand_values["DEMAND_MU"]
    demand_mi_tot = demand_values["DEMAND_MI"]
    
    #--------------------------------------------------------------------------
    # stating variables to store the surplusses/deficits of areas after each
    # iteration
    surplus_co = 0.0
    surplus_in = 0.0
    surplus_mi = 0.0
    surplus_mu = 0.0
    surplus_si = 0.0
    
    year_range = final_year - initial_year
        
    # creating a field that will be a list of the sorted possible uses of each
    # parcel, from best to worst possible use
    gdf["ORDER_POT"] = None
    
    # state a variable that will store all the developed parcels on each iteration
    total_developed = []
    # setting variables to store the ID of the parcels that develop each use
    develop_co_2 = []
    develop_in_2 = []
    develop_si_2 = []
    develop_mu_2 = []
    develop_mi_2 = []
        
    #--------------------------------------------------------------------------
    # setting restriccions related to scenario 2 characteristics
    
    corridor_restriction = []
    
    if "PR_CORR" in gdf.columns and evaluation_column == "2050":
        for index, row in gdf.iterrows():
            if row["PR_CORR"] == 0:
                if row[use_column] != "military" and\
                     row[use_column] != "green_zones":
                        
                    gdf.loc[index, "DEVELOP"] = "vacant"
                    corridor_restriction.append(index)
    
    #--------------------------------------------------------------------------
    # setting restriccions related to scenario 3 characteristics
    
    ls_loss = []
    ls_develop_loss = []
    
    if evaluation_column == "USE_2050":
        
        total_loss = 3893556.81 # loss of 2018-2050 period
        
        gdf = gdf.sort_values("LP", ascending = False)
        ls_loss = gdf["ID"].tolist()
        gdf = gdf.sort_values("ID", ascending = True)
        
    else:
        total_loss = 0
        
    #--------------------------------------------------------------------------
    
    writeLog("Executing the simulation...", outFile)
    
    # execute the simulation, iterating over the range of years using the cicle
    # value
    while initial_year < final_year:
        
        # fix iteration value to adjust to the year value on last iteration
        if final_year - initial_year < cicle:
            iteration = final_year - initial_year

        else:
            iteration = cicle
        
        # iteration range on each iteration
        iteration_range = str(initial_year) + " - " + str(initial_year + iteration)
        writeLog("\nIteration (Year): " + iteration_range, outFile)
        
        writeLog("\nUpdating the attraction values for each parcel...", outFile)
        
        # if there is any developed parcel 
        if total_developed:

            if develop_co_2:
                gdf_sj = update_sj_use_changes(gdf_sj, simulation_column,
                                               "commerce_utilities", develop_co_2)

            if develop_si_2:
                gdf_sj = update_sj_use_changes(gdf_sj, simulation_column,
                                               "single_family", develop_si_2)

            if develop_in_2:
                gdf_sj = update_sj_use_changes(gdf_sj, simulation_column,
                                               "industrial", develop_in_2)
                
            if develop_mu_2:
                gdf_sj = update_sj_use_changes(gdf_sj, simulation_column,
                                               "multi_family", develop_mu_2)
                
            if develop_mi_2:
                gdf_sj = update_sj_use_changes(gdf_sj, simulation_column,
                                               "mixed", develop_mi_2)
            
            # compute the attraction values just of the parcels whose 
            # neighbours changed their uses
            gdf = calc_all_attraction (gdf, gdf_sj, df_coef, uses_list, simulation_column, dist)
            gdf = gdf.sort_values("ID", ascending = True)
            
        # udpdate FRC values
        
        gdf = frc_definer(gdf, frc, uses_list, simulation_column)

        #writeLog("Standarizing the values...\n", outFile)   
        
        # normalize the attraction values using the second method
        #gdf = norm2 (gdf, uses_list)

        #writeLog("Done.\n", outFile)
        
        #gdf_sj.loc[(gdf_sj[simulation_column + "_1"] == neighbour_use), coef_names[i] + "_" + abrv_o] = coefs[i]
        
        # set attraction of uses that not included in the list to 0
        for use in uses_list:
            
            abrv = use[0:2].upper()
            gdf.loc[~gdf[use_column].isin(uses_to_change_list), "ATR_" + abrv + "_UNST"] = 0
        
        #----------------------------------------------------------------------
        
        # compute the potential of transition of each parcel
        writeLog("Calculating the potential uses values for each parcel...", outFile)
        
        for use in uses_list:
            
            abrv = use[0:2].upper()
            
            if use_column in ["USE_1986", "USE_2002"]:
                suitability = gdf["SLOPE"].values
                
            elif use_column == "USE_2018":
                suitability = gdf["S_" + abrv].values

            gdf["POT_" + abrv] = potentials(gdf["DIST"].values,
                                            gdf["ATR_" + abrv + "_UNST"].values, 
                                            gdf["FRC_" + abrv].values,
                                            suitability,
                                            gdf["ZONIF"].values,
                                            alfa)
            
        gdf["ORDER_POT"] = gdf.apply(lambda row: get_order_pot(row, uses_list), axis = 1)
        
        writeLog("Done.\n", outFile)
        
        #----------------------------------------------------------------------
        
        # calculate the demand to be supplied on each iteration
        writeLog("Calculating the demand for each use on this iteration...", outFile)            
        demand_co = ((demand_co_tot/year_range) * iteration) + surplus_co
        demand_in = ((demand_in_tot/year_range) * iteration) + surplus_in
        demand_si = ((demand_si_tot/year_range) * iteration) + surplus_si
        demand_mu = ((demand_mu_tot/year_range) * iteration) + surplus_mu
        demand_mi = ((demand_mi_tot/year_range) * iteration) + surplus_mi
        loss = ((total_loss/year_range) * iteration)
        
        # setting the margins to surpass the demand
        medium_size_co = 5093.625206 * iteration
        medium_size_in = 3633.747749 * iteration
        medium_size_si = 563.5813978 * iteration
        medium_size_mu = 882.0577796 * iteration
        medium_size_mi = 759.7860664 * iteration
        
        writeLog("Done.\n", outFile)
        
        # setting variables to store the area developed on each iteration
        area_co = 0.0
        area_in = 0.0
        area_si = 0.0
        area_mu = 0.0
        area_mi = 0.0
        loss_area = 0.0
        
        # setting variables to store the ID of the parcels that develop each use
        develop_co_2 = []
        develop_in_2 = []
        develop_si_2 = []
        develop_mu_2 = []
        develop_mi_2 = []
        
        # resetting the value of total parcels developped on current iteration
        total_developed = []
        
        # check the demand values to decide to build or not
        # -------------------------
        
        if demand_co_tot > 0:
            build_co = True
        else:
            build_co = False
            
        # -------------------------
        
        if demand_in_tot > 0:
            build_in = True
        else:
            build_in = False
            
        # -------------------------
            
        if demand_si_tot > 0:
            build_si = True
        else:
            build_si = False
            
        # -------------------------
        
        if demand_mu_tot > 0:
            build_mu = True
        else:
            build_mu = False
            
        # -------------------------
            
        if demand_mi_tot > 0:
            build_mi = True
        else:
            build_mi = False
            
        # -------------------------
            
        if total_loss > 0:
            compute_loss = True
        else:
            compute_loss = False
    
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        writeLog("Executing the simulation...", outFile)

        # if there is loss, compute them
        while compute_loss == True:
        
                parcel = ls_loss[0]
                ls_loss.pop(0)
                
                if len(ls_loss) > 0:
        
                    parcel_area = gdf.loc[parcel, "AREA"]
        
                    if ((loss_area + parcel_area) < loss):
                        loss_area += parcel_area
                        ls_develop_loss.append(parcel)
                        total_developed.append(parcel)
                        
                    else:
                        compute_loss = False
                        break
                else:
                    break
        
        #----------------------------------------------------------------------
        
        # for each order priority (the order_pot field of each parcel) evaluate 
        # in descending order the potentials of each parcel. First check the 
        # best possible use for each parcel and use thas ones to develop. 
        # If there is none for a certain use, check the second priority order, 
        # to develop the remaining possible parcels that can fit better on 
        # the demand requirements
        for order in range(0,5):
                      
            writeLog("\nFinding the potential parcels to be developed based on" +\
                     " the current priority (" + str(order) + ") [0 = best for" +\
                     " that use, 1 = second best, and so on] for each use.\n", outFile)
                
            # candidates of each use to develop
            develop_co = []
            develop_in = []
            develop_si = []
            develop_mu = []
            develop_mi = []
            
            # value of potencial of each candidate
            potential_co = []
            potential_in = []
            potential_si = []
            potential_mu = []
            potential_mi = []
            
            # number of candidates of each use and the total
            num_developed_co = len(develop_co_2)
            num_developed_in = len(develop_in_2)
            num_developed_si = len(develop_si_2)
            num_developed_mu = len(develop_mu_2)
            num_developed_mi = len(develop_mi_2)
            num_developed_total = len(total_developed)
            
            # choose the parcels that fits within the specifications previously
            # mentioned (using order priority to select for each use the 
            # parcels that have that use as theis best)
            for row in gdf.itertuples():
                index = row.__getattribute__("ID")
                order_pot = row.__getattribute__("ORDER_POT")
                row_sim_use = row.__getattribute__("SIM_USE")
                
                pot_co = row.__getattribute__("POT_CO")
                pot_in = row.__getattribute__("POT_IN")
                pot_si = row.__getattribute__("POT_SI")
                pot_mu = row.__getattribute__("POT_MU")
                pot_mi = row.__getattribute__("POT_MI")

                if order_pot and len(order_pot) > order and \
                (index not in ls_develop_loss) and\
                (index not in corridor_restriction):
                    
                    if (order_pot[order] == "commerce_utilities") and\
                        (row_sim_use != order_pot[order]) and\
                        (pot_co != 0):
                        develop_co.append(index)
                        potential_co.append(pot_co)
                        
                    if (order_pot[order] == "industrial") and\
                        (row_sim_use != order_pot[order]) and\
                        (pot_in != 0):
                        develop_in.append(index)
                        potential_in.append(pot_in)
                        
                    if (order_pot[order] == "single_family") and\
                        (row_sim_use != order_pot[order]) and\
                        (pot_si != 0):
                        develop_si.append(index)
                        potential_si.append(pot_si)
                        
                    if (order_pot[order] == "multi_family") and\
                        (row_sim_use != order_pot[order]) and\
                        (pot_mu != 0):
                        develop_mu.append(index)
                        potential_mu.append(pot_mu)
                        
                    if (order_pot[order] == "mixed") and\
                        (row_sim_use != order_pot[order]) and\
                        (pot_mi != 0):
                        develop_mi.append(index)
                        potential_mi.append(pot_mi)
                            
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            # Aplication of the transition rules. Are the same for each use so 
            # only one will be explained

            # DEVELOPMENT OF COMMERCE AND UTILITIES
            while build_co == True:
                # if there is no parcels that have their best uses on current
                # priority, wait to the next order priority
                if len(develop_co) == 0:
                    
                    writeLog("There is no parcel whose max potential fits" +\
                             " current order priority (" + str(order) +\
                             ") for commerce and utilities. Will look for best parcels" +\
                             " to develop on next priority order.", outFile)

                    break
                
                else:
                    # choose the max potential value of the list of candidate
                    # parcels
                    x = max(potential_co)
                    
                    # look for the ID of the parcel
                    x_index = potential_co.index(x)
                    parcel = develop_co[x_index]
                    
                    # remove it from the lists
                    potential_co.remove(x)
                    develop_co.remove(parcel)
                    
                    # if the parcel haven´t been already developed on this
                    # iteration, select its area value
                    if parcel not in total_developed:
                        parcel_area = gdf.loc[parcel, "AREA"]

                        # if the current built area plus the area of the evaluated
                        # parcel do not exceed the demand value, use it to develop
                        # and store it in the pertinent lists
                        if ((area_co + parcel_area) < demand_co):
                            area_co += parcel_area
                            develop_co_2.append(parcel)
                            total_developed.append(parcel)
                        
                        # if the current area plus the area of the parcel is 
                        # equal or exceeds the the demand but in lesser
                        # amount that the medium  size plus demand, 
                        # build it and interrupt the building process
                        elif ((area_co + parcel_area) < (demand_co + medium_size_co))\
                            or ((area_co + parcel_area) == demand_co):
                                
                            area_co += parcel_area
                            develop_co_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_co = demand_co - area_co #compute the surplus
                            build_co = False
                            writeLog("The demand of commerce and facilities" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_co) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                        
                        # if there is no build area yet, but the area of the 
                        # parcel already exceeds the demand plus the medium 
                        # size threshold, build it and interrupt the process
                        elif (area_co == 0) and (parcel_area > (demand_co + medium_size_co)):
                            
                            area_co += parcel_area
                            develop_co_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_co = demand_co - area_co #compute the surplus
                            build_co = False
                            writeLog("The firts parcel with the higest potential" +\
                                     " to develop commerce and facilities is" +\
                                     " is bigger than demand. The demand will" +\
                                     " accumulate. " + str(surplus_co) +\
                                     " will be used " + " on next iteration.\n", outFile)
                            break
                        
                        # in any other case (I.E. not the first parcel too big)
                        # interrupt the process, and calculate the deficit
                        else:
                            surplus_co = demand_co - area_co #compute the deficit
                            build_co = False
                            writeLog("The demand of commerce and facilities" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_co) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                          
            #------------------------------------------------------------------

            # DEVELOPMENT OF INDUSTRIAL
            while build_in == True:
                
                if len(develop_in) == 0:
                    
                    writeLog("There is no parcel whose max potential fits" +\
                             " current order priority (" + str(order) +\
                             ") for industrial use. Will look for best parcels" +\
                             " to develop on next priority order.", outFile)

                    break

                else:
                    x = max(potential_in)
                    x_index = potential_in.index(x)
                    parcel = develop_in[x_index]
                    potential_in.remove(x)
                    develop_in.remove(parcel)
                    
                    if parcel not in total_developed:
                        parcel_area = gdf.loc[parcel, "AREA"]

                        if ((area_in + parcel_area) < demand_in):
                            area_in += parcel_area
                            develop_in_2.append(parcel)
                            total_developed.append(parcel)
                            
                        elif ((area_in + parcel_area) < (demand_in + medium_size_in))\
                            or ((area_in + parcel_area) == demand_in):
                                
                            area_in += parcel_area
                            develop_in_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_in = demand_in - area_in
                            build_in = False
                            writeLog("The demand of industrial use" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_in) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                        
                        elif (area_in == 0) and (parcel_area > (demand_in + medium_size_in)):
                            
                            area_in += parcel_area
                            develop_in_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_in = demand_in - area_in
                            build_in = False
                            writeLog("The first parcel with the higest potential" +\
                                     " to develop industrial use is" +\
                                     " is bigger than demand. The demand will" +\
                                     " accumulate. " + str(surplus_in) +\
                                     " will be used " + " on next iteration.\n", outFile)
                            break
                        
                        else:
                            surplus_in = demand_in - area_in
                            build_in = False
                            writeLog("The demand of industrial use" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_in) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                          
            #------------------------------------------------------------------
            
            # DEVELOPMENT OF SINGLE-FAMILITY RESIDENTIALS
            while build_si == True:
                
                if len(develop_si) == 0:
                    
                    writeLog("There is no parcel whose max potential fits" +\
                             " current order priority (" + str(order) +\
                             ") for single-family residentials. Will look for best parcels" +\
                             " to develop on next priority order.", outFile)

                    break
                
                else:
                    x = max(potential_si)
                    x_index = potential_si.index(x)
                    parcel = develop_si[x_index]
                    potential_si.remove(x)
                    develop_si.remove(parcel)
                    
                    if parcel not in total_developed:
                        parcel_area = gdf.loc[parcel, "AREA"]

                        if ((area_si + parcel_area) < demand_si):
                            area_si += parcel_area
                            develop_si_2.append(parcel)
                            total_developed.append(parcel)
                            
                        elif ((area_si + parcel_area) < (demand_si + medium_size_si))\
                            or ((area_si + parcel_area) == demand_si):
                                
                            area_si += parcel_area
                            develop_si_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_si = demand_si - area_si
                            build_si = False
                            writeLog("The demand of single-family residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_si) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                        
                        elif (area_si == 0) and (parcel_area > (demand_si + medium_size_si)):
                            
                            area_si += parcel_area
                            develop_si_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_si = demand_si - area_si
                            build_si = False
                            writeLog("The first parcel with the higest potential" +\
                                     " to develop single-family residential is" +\
                                     " is bigger than demand. The demand will" +\
                                     " accumulate. " + str(surplus_si) +\
                                     " will be used " + " on next iteration.\n", outFile)
                            break
                        
                        else:
                            surplus_si = demand_si - area_si
                            build_si = False
                            writeLog("The demand of single-family residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_si) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
            
            #------------------------------------------------------------------
            
            # DEVELOPMENT OF MULTI-FAMILITY RESIDENTIALS
            while build_mu == True:
                
                if len(develop_mu) == 0:
                    
                    writeLog("There is no parcel whose max potential fits" +\
                             " current order priority (" + str(order) +\
                             ") for multi-family residentials. Will look for best parcels" +\
                             " to develop on next priority order.", outFile)

                    break
                
                else:
                    x = max(potential_mu)
                    x_index = potential_mu.index(x)
                    parcel = develop_mu[x_index]
                    potential_mu.remove(x)
                    develop_mu.remove(parcel)
                    
                    if parcel not in total_developed:
                        parcel_area = gdf.loc[parcel, "AREA"]

                        if area_mu + parcel_area < demand_mu:
                            area_mu += parcel_area
                            develop_mu_2.append(parcel)
                            total_developed.append(parcel)
                            
                        elif ((area_mu + parcel_area) < (demand_mu + medium_size_mu))\
                            or ((area_mu + parcel_area) == demand_mu):
                                
                            area_mu += parcel_area
                            develop_mu_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_mu = demand_mu - area_mu
                            build_mu = False
                            writeLog("The demand of multi-family residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_mu) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                        
                        elif (area_mu == 0) and (parcel_area > (demand_mu + medium_size_mu)):
                            
                            area_mu += parcel_area
                            develop_mu_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_mu = demand_mu - area_mu
                            build_mu = False
                            writeLog("The first parcel with the higest potential" +\
                                     " to develop multi-family residential is" +\
                                     " is bigger than demand. The demand will" +\
                                     " accumulate. " + str(surplus_mu) +\
                                     " will be used " + " on next iteration.\n", outFile)
                            break
                        
                        else:
                            surplus_mu = demand_mu - area_mu
                            build_mu = False
                            writeLog("The demand of multi-family residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_mu) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                          
            #------------------------------------------------------------------
            
            # DEVELOPMENT OF MIXED RESIDENTIALS
            while build_mi == True:
                
                if len(develop_mi) == 0:

                    writeLog("There is no parcel whose max potential fits" +\
                             " current order priority (" + str(order) +\
                             ") for mixed residentials. Will look for best parcels" +\
                             " to develop on next priority order.", outFile)

                    break
                
                else:
                    x = max(potential_mi)
                    x_index = potential_mi.index(x)
                    parcel = develop_mi[x_index]
                    potential_mi.remove(x)
                    develop_mi.remove(parcel)
                    
                    if parcel not in total_developed:
                        parcel_area = gdf.loc[parcel, "AREA"]

                        if ((area_mi + parcel_area) < demand_mi):
                            area_mi += parcel_area
                            develop_mi_2.append(parcel)
                            total_developed.append(parcel)
                            
                        elif ((area_mi + parcel_area) < (demand_mi + medium_size_mi))\
                            or ((area_mi + parcel_area) == demand_mi):
                                
                            area_mi += parcel_area
                            develop_mi_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_mi = demand_mi - area_mi
                            build_mi = False
                            writeLog("The demand of mixed residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_mi) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                        
                        elif (area_mi == 0) and (parcel_area > (demand_mi + medium_size_mi)):
                            
                            area_mi += parcel_area
                            develop_mi_2.append(parcel)
                            total_developed.append(parcel)
                            surplus_mi = demand_mi - area_mi
                            build_mi = False
                            writeLog("The firdt parcel with the higest potential" +\
                                     " to develop mixed residential is" +\
                                     " is bigger than demand. The demand will" +\
                                     " accumulate. " + str(surplus_mi) +\
                                     " will be used " + " on next iteration.\n", outFile)
                            break   
                        
                        else:
                            surplus_mi = demand_mi - area_mi
                            build_mi = False
                            writeLog("The demand of mixed residential" +\
                                     " areas has been suplied, with a deficit/surplus" +\
                                     " of " + str(surplus_mi) + " that will be used" +\
                                     " on next iteration.\n", outFile)
                            break
                    
            #------------------------------------------------------------------
            
            writeLog("\n" + ("-")*100, outFile)
                
            writeLog("\nSUMMARY OF THE PARCELS THAT HAS CHANGED THEIR USE " +\
                     "BASED ON THE BEST FIT PRIORITY:\n", outFile) 
                
            writeLog("For the priority order " + str(order) + ", of the " +\
                     " total of " + str(len(total_developed) - num_developed_total) +\
                     " developed parcels, there have been developed:\n", outFile)

            # summary of the development results
            writeLog(str(len(develop_co_2) - num_developed_co) +\
                       " parcels destined to " + "commerce and facilities.", outFile)
            writeLog(str(len(develop_in_2) - num_developed_in) +\
                       " parcels destined to " + "industrial use.", outFile)
            writeLog(str(len(develop_si_2) - num_developed_si) +\
                       " parcels destined to " + "single-family residentials.", outFile)            
            writeLog(str(len(develop_mu_2) - num_developed_mu) +\
                       " parcels destined to " + "multi-family residentials.", outFile)
            writeLog(str(len(develop_mi_2) - num_developed_mi) +\
                       " parcels destined to " + "mixed residentials.", outFile)   

         
            writeLog("\n" + ("-")*100, outFile)
            
            # if all demand has been supplied or all order priorities have 
            # been evaluated, show finish message
            if build_co == False and build_in == False and build_si == False\
               and build_mu == False and build_mi == False:
                   
                   
                   writeLog("\nThe demand of all uses has been suplied, with " +\
                            "the order " + str(order) + " as the last " +\
                            "priority used. Showing the results.", outFile)
                   break
        #----------------------------------------------------------------------    
             
        # update the new use and the iteration range on which each parcel
        # changed its use 
        for s in develop_co_2:
            gdf.loc[s, "DEVELOP"] = "commerce_utilities"
            gdf.loc[s, "ITERATION"] = iteration_range

        for s in develop_in_2:
            gdf.loc[s,"DEVELOP"] = "industrial"
            gdf.loc[s,"ITERATION"] = iteration_range

        for s in develop_si_2:
            gdf.loc[s,"DEVELOP"] = "single_family"
            gdf.loc[s,"ITERATION"] = iteration_range

        for s in develop_mu_2:
            gdf.loc[s,"DEVELOP"] = "multi_family"
            gdf.loc[s,"ITERATION"] = iteration_range

        for s in develop_mi_2:
            gdf.loc[s,"DEVELOP"] = "mixed"
            gdf.loc[s,"ITERATION"] = iteration_range
            
        if ls_develop_loss:
            for s in ls_develop_loss:
                gdf.loc[s,"DEVELOP"] = "vacant"
                gdf.loc[s,"ITERATION"] = iteration_range
            
        writeLog("\n" + ("-")*100, outFile)
        
        # Show final statistics
        writeLog("\nITERATION " + iteration_range + ":\n", outFile)
        writeLog("AREA DESTINED TO COMMERCE AND FACILITIES: " + str(area_co), outFile)
        writeLog("Number of parcels: " + str(len(develop_co_2)) + "\n", outFile)
        writeLog("AREA DESTINED TO USO INDUSTRIAL USE: " + str(area_in), outFile)
        writeLog("Number of parcels: " + str(len(develop_in_2)) + "\n", outFile)
        writeLog("AREA DESTINED TO SINGLE-FAMILY RESIDENTIALS: " + str(area_si), outFile)
        writeLog("Number of parcels: " + str(len(develop_si_2)) + "\n", outFile)        
        writeLog("AREA DESTINED TO MULTI-FAMILY RESIDENTIALS: " + str(area_mu), outFile)
        writeLog("Number of parcels: " + str(len(develop_mu_2)) + "\n", outFile)        
        writeLog("AREA DESTINED TO USO MIXED RESIDENTIALS: " + str(area_mi), outFile)
        writeLog("Number of parcels: " + str(len(develop_mi_2)) + "\n", outFile)
        writeLog("\n" + ("-")*100, outFile)
        
        # if there is loss to be computed
        if ls_develop_loss:
            writeLog("AREA LOST: " + str(loss_area), outFile)
            writeLog("Number of parcels: " + str(len(ls_develop_loss)) + "\n", outFile)  
        
        writeLog("\n" + ("-")*100, outFile)
        
        # evaluate the conversion. If a parcel is conveted, its a net loss of
        # that use for that iteration, and it counts as a deficit, that has to
        # be suplied in the next iteration
        for desarrolla in total_developed:
            uso_d = gdf.loc[desarrolla, "SIM_USE"]
            area_d = gdf.loc[desarrolla, "AREA"]
            
            if uso_d == "commerce_utilities":
                surplus_co += area_d
            if uso_d == "industrial":
                surplus_in += area_d
            if uso_d == "single_family":
                surplus_si += area_d
            if uso_d == "multi_family":
                surplus_mu += area_d
            if uso_d == "mixed":
                surplus_mi += area_d
                
        writeLog("\nLOST AREA FOR EACH USE (WHICH INCLUDE DEFICIT/SURPLUS" +\
                 " RELATED TO THE FULFILMENT OF DEMAND REQUIREMENTS):\n", outFile)
        
        writeLog("AREA OF COMMERCE AND FACILITIES: " + str(surplus_co), outFile)
        writeLog("AREA OF USO INDUSTRIAL USE: " + str(surplus_in), outFile)
        writeLog("AREA OF SINGLE-FAMILY RESIDENTIALS: " + str(surplus_si), outFile)
        writeLog("AREA OF MULTI-FAMILY RESIDENTIALS: " + str(surplus_mu), outFile)
        writeLog("AREA OF USO MIXED RESIDENTIALS: " + str(surplus_mi), outFile)
        
        # applying a similar but simpler rules as the previous ones to the last
        # iteration. Since the will not be next iteration, the deficits will
        # not be able to supply, so for this last one, an exhaustion GDF is
        # generated, where there is only vacant parcels. The parcels chosen 
        # this time are just the right area to ensure that the final loss
        # will be under the medium size parcel of each use
        if (initial_year + iteration) == final_year:
            
            writeLog("\n" + ("*")*100, outFile)
            
            writeLog("\nLast iteration reached, so that the remaining area" +\
                     " that has not been develop yep has to do it to reach" +\
                     " threshold established for each use. Any parcel that" +\
                     " fits in each order priority and that dont surpass remaining" +\
                     " amount of area lost will be used to develop.\n", outFile)
            
            if demand_co_tot > 0:
                build_co = True
            else:
                build_co = False
                
            # -------------------------
            
            if demand_in_tot > 0:
                build_in = True
            else:
                build_in = False
                
            # -------------------------
                
            if demand_si_tot > 0:
                build_si = True
            else:
                build_si = False
                
            # -------------------------
            
            if demand_mu_tot > 0:
                build_mu = True
            else:
                build_mu = False
                
            # -------------------------
                
            if demand_mi_tot > 0:
                build_mi = True
            else:
                build_mi = False
                
            # -------------------------
                
            if total_loss > 0:
                compute_loss = True
            else:
                compute_loss = False
                    
            # generate a GDF with only vacant land
            gdf_exhaust = gdf[gdf["SIM_USE"] == "vacant"]
            
            develop_co_2 = []
            develop_in_2 = []
            develop_si_2 = []
            develop_mu_2 = []
            develop_mi_2 = []
        
            total_developed2 = []
            
            # same rules as before
            for order in range(0,5):
                                    
                develop_co = []
                develop_in = []
                develop_si = []
                develop_mu = []
                develop_mi = []
                
                potential_co = []
                potential_in = []
                potential_si = []
                potential_mu = []
                potential_mi = []
                        
                for row in gdf_exhaust.itertuples():
                    index = row.__getattribute__("ID")
                    order_pot = row.__getattribute__("ORDER_POT")
                    row_sim_use = row.__getattribute__("SIM_USE")
                    
                    pot_co = row.__getattribute__("POT_CO")
                    pot_in = row.__getattribute__("POT_IN")
                    pot_si = row.__getattribute__("POT_SI")
                    pot_mu = row.__getattribute__("POT_MU")
                    pot_mi = row.__getattribute__("POT_MI")
    
                    if order_pot and len(order_pot) > order and \
                    (index not in ls_develop_loss) and\
                    (index not in corridor_restriction):
                        
                        if (order_pot[order] == "commerce_utilities") and\
                            (row_sim_use != order_pot[order]) and\
                            (pot_co != 0):
                            develop_co.append(index)
                            potential_co.append(pot_co)
                            
                        if (order_pot[order] == "industrial") and\
                            (row_sim_use != order_pot[order]) and\
                            (pot_in != 0):
                            develop_in.append(index)
                            potential_in.append(pot_in)
                            
                        if (order_pot[order] == "single_family") and\
                            (row_sim_use != order_pot[order]) and\
                            (pot_si != 0):
                            develop_si.append(index)
                            potential_si.append(pot_si)
                            
                        if (order_pot[order] == "multi_family") and\
                            (row_sim_use != order_pot[order]) and\
                            (pot_mu != 0):
                            develop_mu.append(index)
                            potential_mu.append(pot_mu)
                            
                        if (order_pot[order] == "mixed") and\
                            (row_sim_use != order_pot[order]) and\
                            (pot_mi != 0):
                            develop_mi.append(index)
                            potential_mi.append(pot_mi)
                            
                #--------------------------------------------------------------
                #--------------------------------------------------------------
                
                # simpler transition rules, just using parcels that will not 
                # exceed demans, util the medium size parcel per use is 
                # achieved
                if build_co == True:
                    while surplus_co > (medium_size_co / iteration):
                        
                        if len(develop_co) == 0:
                            break
                        else:
                            x = max(potential_co)
                            x_index = potential_co.index(x)
                            parcel = develop_co[x_index]
                            potential_co.remove(x)
                            develop_co.remove(parcel)
                            
                            area = gdf_exhaust.loc[parcel, "AREA"]
                            
                            if (parcel not in total_developed2) and (area < surplus_co):
                                
                                surplus_co -= area
                                develop_co_2.append(parcel)
                                total_developed2.append(parcel)
                                
                if build_in == True:
                    while surplus_in > (medium_size_in / iteration):
                        
                        if len(develop_in) == 0:
                            break
                        else:
                            x = max(potential_in)
                            x_index = potential_in.index(x)
                            parcel = develop_in[x_index]
                            potential_in.remove(x)
                            develop_in.remove(parcel)
                            
                            area = gdf_exhaust.loc[parcel, "AREA"]
                            
                            if (parcel not in total_developed2) and (area < surplus_in):
                                
                                surplus_in -= area
                                develop_in_2.append(parcel)
                                total_developed2.append(parcel)  
                                
                if build_si == True:
                    while surplus_si > (medium_size_si / iteration):
                        
                        if len(develop_si) == 0:
                            break
                        else:
                            x = max(potential_si)
                            x_index = potential_si.index(x)
                            parcel = develop_si[x_index]
                            potential_si.remove(x)
                            develop_si.remove(parcel)
                            
                            area = gdf_exhaust.loc[parcel, "AREA"]
                            
                            if (parcel not in total_developed2) and (area < surplus_si):
                                
                                surplus_si -= area
                                develop_si_2.append(parcel)
                                total_developed2.append(parcel)                

                if build_mu == True:
                    while surplus_mu > (medium_size_mu / iteration):
                        
                        if len(develop_mu) == 0:
                            break
                        else:
                            x = max(potential_mu)
                            x_index = potential_mu.index(x)
                            parcel = develop_mu[x_index]
                            potential_mu.remove(x)
                            develop_mu.remove(parcel)
                            
                            area = gdf_exhaust.loc[parcel, "AREA"]
                            
                            if (parcel not in total_developed2) and (area < surplus_mu):
                                
                                surplus_mu -= area
                                develop_mu_2.append(parcel)
                                total_developed2.append(parcel)

                if build_mi == True:
                    while surplus_mi > (medium_size_mi / iteration):
                        
                        if len(develop_mi) == 0:
                            break
                        else:
                            x = max(potential_mi)
                            x_index = potential_mi.index(x)
                            parcel = develop_mi[x_index]
                            potential_mi.remove(x)
                            develop_mi.remove(parcel)
                            
                            area = gdf_exhaust.loc[parcel, "AREA"]
                            
                            if (parcel not in total_developed2) and (area < surplus_mi):
                                
                                surplus_mi -= area
                                develop_mi_2.append(parcel)
                                total_developed2.append(parcel)
                
            # update the last parcels with their new uses, and mark them
            for s in develop_co_2:
                gdf.loc[s, "DEVELOP"] = "commerce_utilities"
                gdf.loc[s, "ITERATION"] = iteration_range + (" (*)")
    
            for s in develop_in_2:
                gdf.loc[s,"DEVELOP"] = "industrial"
                gdf.loc[s,"ITERATION"] = iteration_range + (" (*)")
    
            for s in develop_si_2:
                gdf.loc[s,"DEVELOP"] = "single_family"
                gdf.loc[s,"ITERATION"] = iteration_range + (" (*)")
    
            for s in develop_mu_2:
                gdf.loc[s,"DEVELOP"] = "multi_family"
                gdf.loc[s,"ITERATION"] = iteration_range + (" (*)")
    
            for s in develop_mi_2:
                gdf.loc[s,"DEVELOP"] = "mixed"
                gdf.loc[s,"ITERATION"] = iteration_range + (" (*)")
                
            writeLog("Total extra of " + str(len(total_developed2)) +\
                       " parcels has been developed to ensure minimum total" +\
                       " demand loss.\n" , outFile)
                
            writeLog("THE FINAL LOST AREA FOR EACH USE IS:\n", outFile)
        
            writeLog("AREA OF COMMERCE AND FACILITIES: " + str(surplus_co), outFile)
            writeLog("AREA OF USO INDUSTRIAL USE: " + str(surplus_in), outFile)
            writeLog("AREA OF SINGLE-FAMILY RESIDENTIALS: " + str(surplus_si), outFile)
            writeLog("AREA OF MULTI-FAMILY RESIDENTIALS: " + str(surplus_mu), outFile)
            writeLog("AREA OF MIXED RESIDENTIALS: " + str(surplus_mi), outFile)
            
            # extend the total developed with the ones on last iteration
            total_developed.extend(total_developed2)
        
        # update all the parcels that developed
        for n in total_developed:
            gdf.loc[n, "SIM_USE"] = gdf.loc[n, "DEVELOP"]
        
        # next iteration
        initial_year += iteration
        
    # list with the fields that are not needed, and will be deleted
    # before exporting the final shapefile
    unnecesary_fields = ["ATR_CO", "ATR_MU", "ATR_IN", "ATR_SI","ATR_MI",
                         "ATR_CO_UNST", "ATR_MU_UNST", "ATR_IN_UNST",
                         "ATR_SI_UNST","ATR_MI_UNST","POT_CO", "POT_MU",
                         "POT_IN", "POT_SI","POT_MI","ORDER_POT",
                         "FRC_CO", "FRC_MU", "FRC_IN", "FRC_SI","FRC_MI"]

    if "LP" in gdf.columns:
        unnecesary_fields.append("LP")
    elif "PR_CORR" in gdf.columns:
        unnecesary_fields.append("PR_CORR")
        
    gdf = gdf.drop(unnecesary_fields, axis = 1)
    
    return gdf

#------------------------------------------------------------------------------

def accuracy_assessment(gdf, uses_list, evaluation_column):
    '''
    Perform an accuracy assessment of the simulation. Gives the user, producer
    and overall accuracy for each, global and changed parcels and the area and
    number of parcels. With this, gives a final overall metrics: overall PA and
    UA, Figure of Merit and Growth Simulation Accuracy. Rquires just the gdf of
    the simulation, where is the simulated use of each parcel and the workshop
    use. The output is just one structured DF that contains all the metrics. 
    '''
    use_list = deepcopy(uses_list)
    use_list.append("vacant")
    
    # selecting the parcels that changed or should have been changed
    gdf_changes = gdf[(gdf["DYN_BOOL"] == 1) | (gdf["DEVELOP"].isna() == False)]

    # generating DFs for each type of analysis: accuracy of the area for global 
    # and just changes, and accuracy of number of parcels of global and changes
    df_global_area = DataFrame()
    df_global_parcel = DataFrame()
    df_changes_area = DataFrame()
    df_changes_parcel = DataFrame()   
    
    # filling all DFs with zeros
    for k in use_list:
        for m in use_list:
            df_global_area.loc[k, m] = 0
            df_global_parcel.loc[k, m] = 0
            df_changes_area.loc[k, m] = 0
            df_changes_parcel.loc[k, m] = 0

    # iterate over the input gdf to fill the DFs of the global accuracy. For
    # each parcel check the combination of simulated and workshopped use and 
    # fill the DFs accordingly 
    for index, row in gdf.iterrows():
        k = row["SIM_USE"] # simulated use 
        m = row[evaluation_column] # workshop use
        
        if (k in use_list) and (m in use_list):
    
            df_global_area.loc[k, m] += row["AREA"]
            df_global_parcel.loc[k, m] += 1
    
    # same process as the previous one but using the GDF with only the parcels
    # that represent any kind of change
    for index, row in gdf_changes.iterrows():
        k = row["SIM_USE"] 
        m = row[evaluation_column] 
        
        if (k in use_list) and (m in use_list):
    
            df_changes_area.loc[k, m] += row["AREA"]
            df_changes_parcel.loc[k, m] += 1
               
    # calculate the total area to assess the overall accuracy
    total_area_g = df_global_area.to_numpy().sum() 
    
    # calculate the total parcel number to assess the overall accuracy
    total_parcel_g = df_global_parcel.to_numpy().sum() 
    
    # calculate the total area to assess the overall accuracy
    total_area_c = df_changes_area.to_numpy().sum() 
    
    # calculate the total parcel number to assess the overall accuracy
    total_parcel_c = df_changes_parcel.to_numpy().sum() 
    
    # add a new row/column to represent the UA and PA and fill it with zeros
    df_global_area.loc["UA (%)", "PA (%)"] = 0
    df_global_parcel.loc["UA (%)", "PA (%)"] = 0
    df_changes_area.loc["UA (%)", "PA (%)"] = 0
    df_changes_parcel.loc["UA (%)", "PA (%)"] = 0
    
    # list with the columns (all DFs have the same, just selecting one)
    df_cols = df_global_area.columns
    
    #--------------------------------------------------------------------------
    
    # calculate the producer accuracy
    # just need to iterate once, using col name to loc the cell of each DF
    for col in df_cols: 
        if col != "PA (%)":
            if nansum(df_global_area[col]) != 0: # nansum treat NaN values as 0
                df_global_area.loc["UA (%)", col] =\
                    round((df_global_area.loc[col, col]/nansum(
                        df_global_area[col].to_numpy()))* 100, 2)
                
            if nansum(df_global_parcel[col]) != 0:
                df_global_parcel.loc["UA (%)", col] =\
                    round((df_global_parcel.loc[col, col]/nansum(
                        df_global_parcel[col].to_numpy()))* 100, 2)
                
            if nansum(df_changes_area[col]) != 0:
                df_changes_area.loc["UA (%)", col] =\
                    round((df_changes_area.loc[col, col]/nansum(
                        df_changes_area[col].to_numpy()))* 100, 2)
                
            if nansum(df_changes_parcel[col]) != 0:
                df_changes_parcel.loc["UA (%)", col] =\
                    round((df_changes_parcel.loc[col, col]/nansum(
                        df_changes_parcel[col].to_numpy()))* 100, 2)              
            
    #--------------------------------------------------------------------------
    
    # calculate the producer accuracy para areas
    # need to iterate 4 times, for each DF
    for index, row in df_global_area.iterrows():
        if index != "UA (%)":
            if nansum(row.to_numpy()) != 0:
                df_global_area.loc[index, "PA (%)"] =\
                    round((row[index]/nansum(row.to_numpy())) * 100, 2)
                
    for index, row in df_global_parcel.iterrows():
        if index != "UA (%)":
            if nansum(row.to_numpy()) != 0:
                df_global_parcel.loc[index, "PA (%)"] =\
                    round((row[index]/nansum(row.to_numpy())) * 100, 2)
                
    for index, row in df_changes_area.iterrows():
        if index != "UA (%)":
            if nansum(row.to_numpy()) != 0:
                df_changes_area.loc[index, "PA (%)"] =\
                    round((row[index]/nansum(row.to_numpy())) * 100, 2)                   

    for index, row in df_changes_parcel.iterrows():
        if index != "UA (%)":
            if nansum(row.to_numpy()) != 0:
                df_changes_parcel.loc[index, "PA (%)"] =\
                    round((row[index]/nansum(row.to_numpy())) * 100, 2)
                
    #--------------------------------------------------------------------------
    
    # define accured area variables
    accured_area_g = 0
    accured_area_c = 0
    
    # compute overall accucary for global areas
    for index, row in df_global_area.iterrows():
        if index != "UA (%)":
            for col in df_cols:
                if col != "PA (%)":
                    if col == index:
                        accured_area_g += row[col]
   
    df_global_area.loc["UA (%)", "PA (%)"] =\
        round((accured_area_g/total_area_g) * 100, 2)
    
    # compute overall accucary for changed parcel areas
    for index, row in df_changes_area.iterrows():
        if index != "UA (%)":
            for col in df_cols:
                if col != "PA (%)":
                    if col == index:
                        accured_area_c += row[col]
   
    df_changes_area.loc["UA (%)", "PA (%)"] =\
        round((accured_area_c/total_area_c) * 100, 2)
    
    #--------------------------------------------------------------------------
    
    # define accured number of parcels variables
    accured_parcel_g = 0
    accured_parcel_c = 0
    
    # compute overall accuracy for global parcels
    for index, row in df_global_parcel.iterrows():
        if index != "UA (%)":
            for col in df_cols:
                if col != "PA (%)":
                    if col == index:
                        accured_parcel_g += row[col]
   
    df_global_parcel.loc["UA (%)", "PA (%)"] =\
        round((accured_parcel_g/total_parcel_g) * 100, 2)
    
    # compute overall accuracy for parcels that changed use
    for index, row in df_changes_parcel.iterrows():
        if index != "UA (%)":
            for col in df_cols:
                if col != "PA (%)":
                    if col == index:
                        accured_parcel_c += row[col]
   
    df_changes_parcel.loc["UA (%)", "PA (%)"] =\
        round((accured_parcel_c/total_parcel_c) * 100, 2)
        
    #--------------------------------------------------------------------------
    
    # generate DFs that will serve as titles for each tipe of metric
    df_global_area_TITLE = DataFrame()
    df_global_parcel_TITLE = DataFrame()
    df_changes_area_TITLE = DataFrame()
    df_changes_parcel_TITLE = DataFrame()
    
    # fill them with the titles and separations
    for s in df_global_area.columns:
        df_global_area_TITLE.loc["GLOBAL_AREA", s] = "--------"
        df_global_parcel_TITLE.loc["GLOBAL_PARCEL", s] = "--------"
        df_changes_area_TITLE.loc["CHANGE_AREA", s] = "--------"
        df_changes_parcel_TITLE.loc["CHANGE_PARCEL", s] = "--------"
        
    # unite each DF with its title using concat
    df_global_area = concat([df_global_area_TITLE, df_global_area], axis = 0)
    df_global_parcel = concat([df_global_parcel_TITLE, df_global_parcel], axis = 0)
    df_changes_area = concat([df_changes_area_TITLE, df_changes_area], axis = 0)
    df_changes_parcel = concat([df_changes_parcel_TITLE, df_changes_parcel], axis = 0)
    
    #--------------------------------------------------------------------------
    
    # compute final extra metrics, using SQL query. If any error derived, use
    # DF default method

    A = len(gdf[(gdf["DYN_BOOL"] == 1) & (gdf["DEVELOP"].isna() == True)])
    B = len(gdf[gdf["DEVELOP"] == gdf["USE_2050"]])
    C = len(gdf[(gdf["DYN_BOOL"] == 1) & (gdf["DEVELOP"].isna() == False) &\
                (gdf["DEVELOP"] != gdf["USE_2050"])])
    D = len(gdf[(gdf["DYN_BOOL"] == 0) & (gdf["DEVELOP"].isna() == False)])
    E = len(gdf[(gdf["DYN_BOOL"] == 0) & (gdf["DEVELOP"].isna() == True)])
        
    PA = B/(A+B+C)*100
    UA = B/(B+C+D)*100
    FOM = B/(A+B+C+D)*100
    GSA = (B+C)/(A+B+C)*100
    OA = (B+E)/(A+B+C+D+E)*100
    
    # add to the last DF the extra metrics
    df_changes_parcel.loc["Overall Producer Accuracy: ", "Value(%)"] = round(PA, 2)
    df_changes_parcel.loc["Overall User Accuracy: ", "Value(%)"] = round(UA, 2)
    df_changes_parcel.loc["Figure of Merit: ", "Value(%)"] = round(FOM, 2)
    df_changes_parcel.loc["Growth Simulation Accuracy: ", "Value(%)"] = round(GSA, 2)
    df_changes_parcel.loc["Overall Accuracy: ", "Value(%)"] = round(OA, 2)    
    
    # unite all the DFs to a final one
    result = concat([df_global_area, df_global_parcel, df_changes_area, df_changes_parcel],
                      axis = 0)

    return result

#------------------------------------------------------------------------------

def build_input_data(outFile, wd, shp, dist_list_regression, uses_list,
                     new_coef, use_column, evaluation_column, att_type):
    
    gdf = read_file(path.join(wd, shp))
    gdf["ID"] = gdf.index
    
    #--------------------------------------------------------------------------
    
    writeLog("Setting working directory in: " + wd + "\n", outFile)
    chdir(wd)
    
    #--------------------------------------------------------------------------
    writeLog("Opening pre-built files..." + "\n", outFile)

    # directory to save the coefficients
    dir_coef = wd + "\\coefficients"
    try: 
        chdir(dir_coef)
    except:
        makedirs(dir_coef)
        chdir(dir_coef)
    
    #--------------------------------------------------------------------------
    curvefit_filename = shp.split(".")[0] + "_coefficients_" + att_type + ".csv"
    
    writeLog("Opening coefficients parameters file..." + "\n", outFile)

    if path.exists(curvefit_filename) and new_coef == False:
        
        d_correlation_study = read_csv(curvefit_filename)
        
    else:
        
        writeLog("Coefficients file has not been generated. Computing "+\
         "coefficients values..." + "\n", outFile)
        
        gdf_data = deepcopy(gdf[["ID", "AREA", use_column, "geometry"]])
        gdf_data = gdf_data.sort_values("ID", ascending = True)
        
        for dist in dist_list_regression:

            writeLog("Neighbours of buffer distance " + str(dist) +\
                     " has not been generated! Generating..." + "\n", outFile)
        
            # run neigbours function
            neighbours_df = neighbours_vectorized(gdf, dist, uses_list,
                                                  use_column)
                        
            neighbours_df = neighbours_df.sort_values("ID", ascending = True)

            #------------------------------------------------------------------
                        
            gdf_data["BFF_AREA_NOTSELF_" + str(dist)] = neighbours_df["TOTAL_N_AREA"]
            gdf_data["BFF_AREA_" + str(dist)] = gdf_data["AREA"] + gdf_data["BFF_AREA_NOTSELF_" + str(dist)]

            for use in uses_list:
                gdf_data["BFF_" + str(dist) + "_AREA_USE_" + use[0:2].upper()] =\
                    neighbours_df["BFF_" + "_AREA_USE_" + use[0:2].upper()]
                    
                gdf_data["BFF_" + str(dist) + "_PROPORTION_" + use[0:2].upper()] =\
                        calc_proportions(gdf_data["BFF_" + str(dist) + "_AREA_USE_" + use[0:2].upper()].values,
                                         gdf_data["BFF_AREA_" + str(dist)].values)
                
                gdf_data["BFF_" + str(dist) + "_PROPORTION_" + use[0:2].upper()] =\
                    gdf_data["BFF_" + str(dist) + "_PROPORTION_" + use[0:2].upper()].fillna(0)
                                
        #----------------------------------------------------------------------

        att_gdf = raw_att_generation(gdf_data, uses_list, use_column,
                                     dist_list_regression, att_type)

        #----------------------------------------------------------------------
        
        writeLog("Finding coefficients values...", outFile)

        # Fit curve generation
        coef_names = ["a", "b", "c", "d", "e"]

        array_x = array(att_gdf.columns[2:], dtype=float)

        d_correlation_study = deepcopy(att_gdf[{"origin_use", "evaluated_use"}])
                        
        # iterate over the current attraction values DF
        for index, row in att_gdf.iterrows():
            
            # Y will be an array with the row values of the read DF, again
            # avoinding the origin and evaluated use 
            array_y = deepcopy(array(row[2:], dtype=float))  

            popt, pcov = curve_fit(f_poly_g3, array_x, array_y, maxfev = 200000, method = "lm")
                
            # add the variables
            for variable in range (0, len(popt)):
                
                d_correlation_study.loc[index, coef_names[variable]] = popt[variable]
            
            # calculate of the predicted y to compute the error
            y_pred = []
                    
            for p in array_x:
                
                valor_pred = deepcopy(f_poly_g3(p, *popt))
                y_pred.append(valor_pred)
            
            # errors computation
            R_SQRT = compute_r2(array_y, y_pred)
            v_MSE = MSE(array_y, y_pred)
            
            # add the error values
            d_correlation_study.loc[index, "R_SQRT"] = R_SQRT
            d_correlation_study.loc[index, "MSE"] = v_MSE
        
        writeLog("Generation of coefficients finished!", outFile)
        
        writeLog("Saving coefficients file...", outFile)        
        chdir(dir_coef)
        d_correlation_study.to_csv(curvefit_filename, index = False)
        
        writeLog("Done", outFile)

        #----------------------------------------------------------------------

        #Plot generation
        
        d_titles = {"commerce_utilities":"Commerce and utilities",
                     "single_family":"Single-family residential",
                      "multi_family": "Multi-family residential",
                      "industrial": "Industrial",
                      "mixed": "Mixed (commercial & residential)",
                      "vacant": "Nonurban"}
        
        d_colors = {"commerce_utilities":"#B53535",
                    "single_family":"#6699CD",
                     "multi_family": "#002673",
                     "industrial": "#704489",
                     "mixed": "#F6C567",
                     "vacant": "#9b9b9b"}

        d_colors_pred = {"commerce_utilities":"#de2323",
                         "single_family":"#43567c",
                         "multi_family": "#001351",
                         "industrial": "#6413c1",
                         "mixed": "#d99d21",
                         "vacant": "#424344"}

        # format to display the data (pair of uses)
        nrow, ncol = 3, 2
        axe1, axe2 = 0, 0

        # subset of plots
        figure, axes = plt.subplots(nrows = nrow, ncols = ncol)

        # iterate over te uses
        for use in uses_list:
            
            df_uso = deepcopy(att_gdf[att_gdf["origin_use"] == use])
            df_uso = df_uso.drop("origin_use", axis = 1)
            df_uso = df_uso.set_index("evaluated_use")
            df_uso_t = df_uso.transpose()
            df_uso_t.index = df_uso_t.index.rename('distancia')
            
            df_uso_t_y_pred = DataFrame()
            dist_values = list(df_uso_t.index)
            lower_range = int(dist_values[0])
            top_range = int(dist_values[-1:][0])
            linspace = np.linspace(lower_range, top_range, top_range - lower_range)
            df_uso_t_y_pred["distance"] = linspace
            df_uso_t_y_pred = df_uso_t_y_pred.set_index("distance")
            
            for use2 in uses_list:

                coefs = d_correlation_study.loc[(d_correlation_study["origin_use"] == use) &\
                                (d_correlation_study["evaluated_use"] == use2)].values.tolist()
                
                coefs = coefs[0][2:6]
            
                y_pred = calc_y_pred(linspace, coefs[0], coefs[1], coefs[2], coefs[3])
                df_uso_t_y_pred[use2] = y_pred 

            for use2 in uses_list:

                axes[axe1, axe2].plot(use2, data = df_uso_t,
                                      color = d_colors[use2],
                                      linewidth = 1.5)
                
            for use2 in uses_list:

                axes[axe1, axe2].plot(use2, data = df_uso_t_y_pred,
                                      color = d_colors_pred[use2],
                                      linewidth = 1.5, linestyle = "--")

            axes[axe1, axe2].set_title(d_titles[use], loc = "left",
                                       fontweight = "bold")
                
            if axe1 < nrow - 1:
                axe1 += 1
            elif axe1 == nrow - 1: 
                axe1 = 0
                axe2 += 1

        y_label = att_type
        for axe in axes.flat:
            axe.set_xlabel('Distance (m)', fontsize = 9, loc = "right")
            axe.set_ylabel(y_label, color = "#D16103", loc = "top")

        # borders of the plots
        for axe in axes.flat:
            axe.spines['right'].set_visible(False)
            axe.spines['top'].set_visible(False)
            axe.spines['bottom'].set_visible(False)
            axe.spines['left'].set_visible(False)

        # plot grids
        for axe in axes.flat:
            axe.grid(color = 'gainsboro', linestyle = '-', linewidth = 0.75)

        figure.legend((uses_list), loc = 8,
                      ncol = 3, bbox_to_anchor=(0.27, -0.14, 0.5, 0.5), 
                      frameon = False, fontsize = 12)
    
        # size
        figure.set_size_inches(8, 6)

        # spacing
        figure.tight_layout(pad = 0, w_pad = 4, h_pad = 2)

        # export the plots
        figure.savefig("attraction_values_" + att_type + ".jpg", quality = 100, dpi = 500,
                       bbox_inches = 'tight')

    return gdf, d_correlation_study

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#SIMULATION
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def set_up_data(outFile, wd, shp, gdf, d_correlation_study, z_default, z_urban,
                z_nonurban, z_protected, dist_list, uses_list,
                dist_list_regression, frc_list, alfa_list, numero_ejecuciones,
                cicle,initial_year, final_year, new_ini_att,
                use_column, evaluation_column, att_type, demand_values):
    
    working_gdf = deepcopy(gdf)
    working_gdf = working_gdf.sort_values("ID", ascending = True)    
    
    # add a field to store the results of the simulation. Use as base the 
    # current use of each parcel
    simulation_column = "SIM_USE"
    working_gdf[simulation_column] = working_gdf[use_column]
        
    # add rows related to the develop use and the iteration when it occurs
    working_gdf["DEVELOP"] = None
    working_gdf["ITERATION"] = None
    
    # normalize the distance to network field
    working_gdf = norm (working_gdf, "SLOPE")
    working_gdf = norm (working_gdf, "DIST")
    
    working_gdf["DYN_BOOL"] = 0
    working_gdf.loc[gdf[use_column] != working_gdf[evaluation_column], "DYN_BOOL"] = 1
        
    # setting the zoning values 
    if z_default == True:
        
        # if no zoning (default = True) a proportion of each type will be
        # calculated and will be established as probability
        zonif_1 = len(gdf[(gdf["DYN_BOOL"] == 1) & (gdf["ZONIF"] == 1)])\
                /len(gdf[gdf["DYN_BOOL"] == 1])
        zonif_2 = len(gdf[(gdf["DYN_BOOL"] == 1) & (gdf["ZONIF"] == 2)])\
                /len(gdf[gdf["DYN_BOOL"] == 1])
        zonif_3 = len(gdf[(gdf["DYN_BOOL"] == 1) & (gdf["ZONIF"] == 3)])\
                /len(gdf[gdf["DYN_BOOL"] == 1])
        
        for index, row in working_gdf.iterrows():
            
            if row["ZONIF"] == 1:
                working_gdf.loc[index, "ZONIF"] = zonif_1
            if row["ZONIF"] == 2:
                working_gdf.loc[index, "ZONIF"] = zonif_2  
            if row["ZONIF"] == 3:
                working_gdf.loc[index, "ZONIF"] = zonif_3
    
    else:
        
        # use custom values of zoning based on users definition
        for index, row in working_gdf.iterrows():
            
            if row["ZONIF"] == 1:
                working_gdf.loc[index, "ZONIF"] = z_urban
            if row["ZONIF"] == 2:
                working_gdf.loc[index, "ZONIF"] = z_nonurban  
            if row["ZONIF"] == 3:
                working_gdf.loc[index, "ZONIF"] = z_protected
    
    #--------------------------------------------------------------------------
    
    # wd to save the results
    wd_result = path.join(wd, "output")
    
    # directory to save the results
    dir_initial_att = wd + "\\initial_attraction"
        
    #----------------------------     
    try:
        chdir(wd_result)
    except:
        makedirs(wd_result)
    #----------------------------
    try: 
        chdir(dir_initial_att)
    except:
        makedirs(dir_initial_att)  
 
    # working directory of the coeficcents of the regression
    df_coef = deepcopy(d_correlation_study)
    
    # iterate over the different buffer distances selected
    for dist in dist_list:
    
        initial_att_filename = shp.split(".")[0] + "_initial_att_" + att_type +\
            "_" + str(dist) + ".csv"
        
        chdir(dir_initial_att)
        
        writeLog("Opening initial attraction file..." + "\n", outFile)

        if path.exists(initial_att_filename) and new_ini_att == False:
            
            d_initial_att = read_csv(initial_att_filename)
            d_initial_att = d_initial_att.sort_values("ID", ascending = True)
            
            # add the attraction values to the WGDF
            for use in uses_list:
                
                abrv = use[0:2].upper()
                working_gdf["ATR_" + abrv + "_UNST"] = d_initial_att["ATR_" + abrv]
                working_gdf["ATR_" + abrv] = working_gdf["ATR_" + abrv + "_UNST"]
                working_gdf["POT_" + abrv] = 0.0
                
            if cicle == (final_year - initial_year):
                gdf_sj = None
            
            else:
                gdf_sj = generate_spatial_join(working_gdf, simulation_column, dist)

        else: 
                
            writeLog("Initial attraction file has not been generated. Computing "+\
             "initial attraction values..." + "\n", outFile)
            
            gdf_sj = generate_spatial_join(working_gdf, simulation_column, dist)
            
            working_gdf = calc_all_attraction (working_gdf, gdf_sj,
                                               df_coef, uses_list,
                                               simulation_column, dist)
    
            # compute the attraction values just of the parcels whose 
            # neighbours changed their uses
            d_initial_att = working_gdf[{"ID", "ATR_SI", "ATR_IN", "ATR_CO", "ATR_MU", "ATR_MI"}]
            d_initial_att = d_initial_att.sort_values("ID", ascending = True)
            
            writeLog("Saving file...", outFile) 
            d_initial_att.to_csv(initial_att_filename, index = False)
            writeLog("Done!", outFile)
            
        chdir(wd_result)

        #----------------------------------------------------------------------

        # iterate over the list of frc user selected
        for frc in frc_list:
            
            # iterate over the list of alfa user selected     
            for i in range (1, numero_ejecuciones + 1):
                
                for alfa in alfa_list:
                    
                    work_gdf_copy = deepcopy(working_gdf)

                    # declaring the name that final output files will have
                    sim_name = shp.split(".")[0] + "_BUFFER_" + str(dist) + "_" + str(frc) + "_" + str(alfa) + "_C" + str(cicle)
                    
                    # differentiating every output with the time it was
                    # generated
                    wd_each_sim = wd_result + "\\" + strftime(sim_name +\
                                                              "_%d-%m-%Y_%H-%M-%S",
                                                              localtime())
                    # create a folder that will contain all the simulation
                    # output files
                    makedirs(wd_each_sim)
                    chdir(wd_each_sim)
    
                    sim = simulation(work_gdf_copy, df_coef, initial_year,
                                     final_year, frc, alfa, cicle, outFile,
                                     use_column, evaluation_column, 
                                     simulation_column, dist, demand_values,
                                     uses_list, gdf_sj)
                    
                    sim = sim.sort_values("ID", ascending = True)
                    
                    writeLog("\nCalculating the confussion matrix and accuracy metrics of the simulation...",
                             outFile)
                    # execute the accuracy assesment
                    conf_matrix = accuracy_assessment(sim, uses_list, evaluation_column)
                    writeLog("Done.\n", outFile)  
                    
                    #----------------------------------------------------------
                    
                    # save the GDF as shapefile
                    sim_result_name = sim_name + ".shp"
                        
                    writeLog("Saving simulation result file: " + sim_result_name,
                             outFile)
                    
                    sim = sim[["ID", "REFCAT", "AREA", use_column, evaluation_column,
                                "SIM_USE", "DEVELOP", "ITERATION", "geometry"]]

                    sim.to_file(sim_result_name)
                    
                    #----------------------------------------------------------
                    # save the accuracy assesment as a excel file
                    
                    conf_matrix_name = sim_name + ".xlsx"
                        
                    writeLog("Saving accuracy file (xlsx): " + conf_matrix_name,
                             outFile)
                    conf_matrix.to_excel(conf_matrix_name)
                    
                    # saving all the configuration used for current 
                    # current simulation on the LOG file
                    
                    #----------------------------------------------------------
                    
                    fmt = "| {{:<{}s}} | {{:>{}s}} | {{:>{}s}} |".format(42, 8, 11)
                    parametros = "\n\n" + "#"*79 + "\n" + "#"*79 + "\n\n" +\
                    "PARAMETERS USED FOR THE SIMULATION:\n\n" +\
                    fmt.format("DESCRIPTION", "VARIABLE", "VALUE") + "\n" +\
                    "-"*71 + "\n" +\
                    fmt.format("Year to simulate", "YYYY", str(final_year)) + "\n" +\
                    fmt.format("Buffer size", "dist", str(dist)) + "\n" +\
                    fmt.format("Resistance to change factor", "frc", str(frc)) + "\n" +\
                    fmt.format("Degree of randomness", "alfa", str(alfa)) + "\n" +\
                    fmt.format("Priority of urban parcels over rustic ones", "urb_prio", "0.98") + "\n" +\
                    fmt.format("Lapse of time that each intern simualting", "", "") + "\n" +\
                    fmt.format("iteration represents(years)", "ciclo", str(cicle)) + "\n\n" +\
                    "#"*79 + "\n" + "#"*79 +  "\n\n"
                                                
                    writeLog(parametros, outFile)
                    
                del sim, sim_name
                
        del work_gdf_copy
                        
    del working_gdf

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    
    # save initial time
    t_start = time()
    
    #--------------------------------------------------------------------------
    # files and directory declaration
    
    wd = "C:\\Users\\Usuario\\OneDrive - Universidad de Alcala\PoC\\scenario_3_v6"
    shp = "scenario_3.shp"
    
    #--------------------------------------------------------------------------
    # create the LOG file
    elLog = strftime(wd + "\LOG_simulation_%d%m%Y_%H%M.log", 
                     localtime())
    outFile = open(elLog, "w")
    
    #--------------------------------------------------------------------------
    # simmulation time properties
    initial_year = 2002
    final_year = 2018
    cicle = 16
    numero_ejecuciones = 1
    
    #--------------------------------------------------------------------------
    # selection of the column that defines the uses of initial year
    
    #use_column = "USE_1986"
    #use_column = "USE_2002" 
    use_column = "USE_2018"
    
    #--------------------------------------------------------------------------
    # selection of the column that defines the uses of final year
    
    #evaluation_column = "USE_2002"
    #evaluation_column = "USE_2018" 
    evaluation_column = "USE_2050"

    #--------------------------------------------------------------------------
    
    demand_values_1986_2002 = {"DEMAND_CO": 2945626.395,
                               "DEMAND_IN": 5049291.415,
                               "DEMAND_SI": 4552546.524,
                               "DEMAND_MU": 1191042.255,
                               "DEMAND_MI": 386495.8035}
     
    demand_values_2002_2018 = {"DEMAND_CO": 2586324.531,
                               "DEMAND_IN": 3912516.708,
                               "DEMAND_SI": 1968408.283,
                               "DEMAND_MU": 1179330.054,
                               "DEMAND_MI": 156180.7106}
    
    demand_values_2018_2050 = {"DEMAND_CO": 1141172.923,
                               "DEMAND_IN": 7176614.766,
                               "DEMAND_SI": 11635502.75,
                               "DEMAND_MU": 677720.3006,
                               "DEMAND_MI": -143402.512}
    
    if use_column == "USE_2018":
        demand_values = demand_values_2018_2050
        
    elif use_column == "USE_2002":
        demand_values = demand_values_2002_2018
        
    elif use_column == "USE_1986":
        demand_values = demand_values_1986_2002
    
    #--------------------------------------------------------------------------
    # list of different values of variables to use on simulations
    
    dist_list = [50]
    #dist_list_regression = [50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    dist_list_regression = [50, 75, 100, 125]
    
    #-------------------------------------------------------¡-------------------
    
    att_type = "vF"
    
    #--------------------------------------------------------------------------
    
    new_coef = True
    new_ini_att = True

    #--------------------------------------------------------------------------

    frc_list = [0]
    
    #alfa_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    alfa_list =  [0]
        
    #--------------------------------------------------------------------------
    # list with the different uses to use on the simmulation
    
    uses_list = ["commerce_utilities", "single_family", "multi_family",
                "industrial", "mixed"]
    
    #--------------------------------------------------------------------------
    # zoning factor options
    
    z_default = True
    z_urban = 2
    z_nonurban = 1
    z_protected = 0
    
    #--------------------------------------------------------------------------

    gdf, d_correlation_study = build_input_data(outFile, wd, shp,dist_list_regression,
                                                uses_list, new_coef, use_column,
                                                evaluation_column, att_type)

    #--------------------------------------------------------------------------

    set_up_data(outFile, wd, shp, gdf, d_correlation_study, z_default, z_urban,
                z_nonurban, z_protected, dist_list, uses_list,
                dist_list_regression, frc_list, alfa_list, numero_ejecuciones,
                cicle, initial_year, final_year, new_ini_att,
                use_column, evaluation_column, att_type, demand_values)

    #--------------------------------------------------------------------------

    # save the final time
    t_finish = time()
    
    # show the execution time elapsed
    t_process = (t_finish - t_start) / 60
    
    writeLog("Process time: " + str(round(t_process, 2)) + "minutes", outFile)
    
    outFile.close()     

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()




            
    






