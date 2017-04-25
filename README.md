# MBAR_sampling_LJ_methane
This repository is designed to compare the MBAR results that are obtained for the Lennard-Jones representation of methane using configurations from 92 states and from just a single state.

For "lj_compare_u0ln_with_u92ln.py" to run you must have the following files and directories:

/data
/simtk.unit-0.2
LJTPstar.npy
LJTPstar.txt
methane.experiment.txt

The primary results can be observed in the parity plots that are generated.

Namely, the enthalpy appears to be consistently overpredicted when configurations are obtained from only a single state.
Likewise, the density is nearly constant when configurations are obtained from only a single state.
The deviations with respect to epsilon and sigma demonstrate systematic errors.
The primary reason for the poor performance is that only a few configurations have appreciable weights for several states.
