# -*- coding: utf-8 -*-
"""
@Author: UAH - TRANSURBAN - Nikolai Shurupov - Ramón Molinero Parejo
@Date: Thu Dec 10 17:15:09 2020
@Version: 2.0
@Description: Disruptive vetorial cellullar automata simulation module.
"""

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
   






