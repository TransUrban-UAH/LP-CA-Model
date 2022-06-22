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
from pandas import DataFrame, read_csv
from os import chdir, makedirs, path
from geopandas import read_file
#from tqdm import tqdm
from copy import deepcopy
import numpy as np
from numpy import array, var

# imports to make plots
from matplotlib import pyplot as plt

# imports to perform regressions
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as MSE

#local imports
from LP_CA_WriteLog import writeLog
from LP_CA_Neighbourhood import neighbours_vectorized
from LP_CA_Attraction import raw_att_generation

###############################################################################
###############################################################################

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

def calc_proportions (numpy_array1: np.array, numpy_array2: np.array):
    
    proportions = numpy_array1/numpy_array2
    
    return proportions

#------------------------------------------------------------------------------

def calc_y_pred (x: np.array, a, b, c, d) -> np.array:
    
    att_array = (a*(x**3)) + (b*(x**2)) + (c*x) + d
    
    return att_array

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


    






