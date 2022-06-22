# -*- coding: utf-8 -*-
"""
@Author: UAH - TRANSURBAN - Nikolai Shurupov - Ramón Molinero Parejo
@Date: Thu Dec 10 17:15:09 2020
@Version: 2.0
@Description: Disruptive vetorial cellullar automata simulation module.

"""

###############################################################################
###############################################################################

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

    






