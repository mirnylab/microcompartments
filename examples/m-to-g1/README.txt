Example code was generated with the following command. Note that using numpolyblocks < default value here results in simulation running only until end of early G1. The simulation took approximately 10 minutes to run on a NVIDIA GeForce RTX 4090 GPU.

python m-to-g1_transition.py npoly=7700 initpolyblocks=7200 numpolyblocks=7800 microcomppath=dag1/microcompsAnaTelo.dat comppath=dag1/comps.dat ctcfpath=dag1/ctcf.dat outpath=examples
