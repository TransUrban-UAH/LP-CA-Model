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
from pandas import read_csv
from os import chdir, makedirs, path
from time import strftime, localtime, time
from copy import deepcopy

#local imports
from LP_CA_Normalization import norm
from LP_CA_WriteLog import writeLog
from LP_CA_Neighbourhood import generate_spatial_join
from LP_CA_Attraction import calc_all_attraction
from LP_CA_Simualtion import simulation
from LP_CA_Accuracy import accuracy_assessment
from LP_CA_Build_input_data import build_input_data

###############################################################################
###############################################################################

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

#------------------------------------------------------------------------------
        
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




            
    






