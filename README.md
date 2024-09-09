# microcompartments
Polymer simulations of compartments and extrusion for modeling the mitosis-to-G1 transition

Reference: biorxiv <...>
Also see: The polychrom library (https://github.com/open2c/polychrom/), a wrapper for the 
OpenMM MD package. Beyond using the polychrom library, the code in this repository used 
examples and methods found in polychrom as a starting point.

There are two simulation codes here:
1) comp_extr - This code is used for equilibrated polymer sims (used in parameter sweeps)
2) m-to-g1 - A code for performing time-calibrated polymer simulations that progress from
   mitotic-like chromosomes to interphase-like chromosomes.
