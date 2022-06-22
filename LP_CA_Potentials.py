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
from copy import deepcopy
import numpy as np
from random import uniform
from math import log

###############################################################################
###############################################################################

def alpha_generator(x, alfa):
    return x + (-log(uniform(0, 1)))**alfa

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







