# LP-CA-Model to simulate disruptive scenarios using vector-based cellular automata

## Summary

The simulation software is divided into 9 main modules. Each module is encharged of some part of the process. Mainly, to carry out the simulation it is needed a vetorial file in shapefile format. This SHP must have the accesibility, suitability and zoning values for each parcel. The process subdivides in: 1) generate working data and 2)perform the simulations. Both prosses execute one after the other, and once the working data has been generated, and unless user don't want to force a new generation, subsequent simulations will use this data, which will result in faster results generation.

## Installation

The scripts have been built under python version 3.8.6 using Spyder as the IDE and Anaconda as the environment manager. All the required packages are
stated in the requirements.txt file.

## How to Run

LP_CA_Main script should be executed after adjusting all the parameters, mainly the year range to simmulate, the demand of each use to develop, buffer size to approximate the regressión ecuations, buffer size to calculate the attraction values, resistance to change facto and, lastly, randomness factor.
**A dataset of the municipality of Alcalá de Henares is given to work as an example of how to perform the whole process. This dataset includes all the necessary data to directly run the simulation.**

## Files

* ``LP_CA_Main.py``: The main script that user should open, modify according its needs and execute.
* ``LP_CA_Build_inup_data.py``: Organize and generate the data needed by the simulation using the rest of the modeles.
* ``LP_CA_Attraction.py``: Module echarged of updating the attraction values of each parcel, using vectorization processes.
* ``LP_CA_Potentials.py``: Module echarged of computing the potential values of each parcel, using vectorization processes.
* ``LP_CA_Simulation.py``: Core module that performs the simulation itself, where the transition rules are established.
* ``LP_CA_Accuracy.py``: Module that perform the accuracy assesment of the simulation results.
* ``LP_CA_Normalization.py``: Module encharged of the normalization of the different data of the input shapefile.
* ``LP_CA_Neighbourhood.py``: Module encharged of the generation of neighbourhood relations, letting different buffer sizes as input.
* ``LP_CA_WriteLog.py``: Small module for the LOG generation.
