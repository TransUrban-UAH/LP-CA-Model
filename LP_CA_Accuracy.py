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
from pandas import DataFrame, concat
#from tqdm import tqdm
from copy import deepcopy
from numpy import nansum

###############################################################################
###############################################################################

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







