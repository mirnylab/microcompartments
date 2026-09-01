# microcompartments
Polymer simulations of compartments and extrusion for modeling the mitosis-to-G1 transition
and, more generally, chromatin compartmentalization with or without loop extrusion. See Refs [1-2].

Also see: The polychrom library (https://github.com/open2c/polychrom/) [3], a wrapper for the 
OpenMM MD package [4]. Beyond using the polychrom library, the code in this repository used 
examples and methods found in polychrom as a starting point.

There are two simulation codes here:
1) *comp_extr* - This code is used for equilibrated polymer sims (used in parameter sweeps)
2) *m-to-g1* - A code for performing time-calibrated polymer simulations that progress from
   mitotic-like chromosomes to interphase-like chromosomes.

Additionally, the examples directory contains data from short, small runs as examples, along with
the commands (with command line options) used to run the simulations.

### Running microcompartment simulations ###
To run steady-state simulations simulations in the compartment/microcompartment configuration used to
model the *Dag1* locus, provide the configuration files as inputs in the command line. e.g.:

python compSim.py comppath=dag1/comps.dat microcomppath=dag1/microcompsAnaTelo.dat

Other parameters can be set/changed by command line inputs as well. For more details:

python compSim.py ?

M-to-G1 simulations of the *Dag1* locus are run similarly (with the addition of an input for CTCF positions):

python m-to-g1_transition.py comppath=dag1/comps.dat microcomppath=dag1/microcompsAnaTelo.dat ctcfpath=dag1/ctcf.dat

### Additional notes ###
Code will run from directory without additional setup (provided that polychrom and OpenMM are also
installed). Simulations were originally run on GPUs on machines running Ubuntu 22.04.5 OS.

## References ##
[1] VY Goel et al.  Dynamics of microcompartment formation at the mitosis-to-G1 transition. *Nat. Struct. Mol Biol.* 32:2614 (2025).

[2] G Wutz, IF Davidson, EJ Banigan et al. PDS5 proteins control genome architecture by limiting the lifetime of cohesin-NIPBL complexes. *Mol. Cell* 86:1614 (2026).

[3] M Imakaev, A Goloborodoko, HB Brandao. polychrom v0.1.0. *Zenodo*: https://zenodo.org/records/3579473 DOI: 10.5281/zenodo.3579472

[4] P Eastman et al. OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials. *J Phys Chem B* 128:109-116 (2023).
