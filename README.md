# microcompartments
Polymer simulations of compartments and extrusion for modeling the mitosis-to-G1 transition

Reference: bioRxiv <...>

Also see: The polychrom library (https://github.com/open2c/polychrom/) [1], a wrapper for the 
OpenMM MD package [2]. Beyond using the polychrom library, the code in this repository used 
examples and methods found in polychrom as a starting point.

There are two simulation codes here:
1) *comp_extr* - This code is used for equilibrated polymer sims (used in parameter sweeps)
2) *m-to-g1* - A code for performing time-calibrated polymer simulations that progress from
   mitotic-like chromosomes to interphase-like chromosomes.

## References
[1] M Imakaev, A Goloborodoko, HB Brandao. polychrom v0.1.0. *Zenodo*: https://zenodo.org/records/3579473 DOI: 10.5281/zenodo.3579472
[2] P Eastman et al. OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials. *J Phys Chem B* 128:109-116 (2023)
