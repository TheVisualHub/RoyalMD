# 👑 Welcome to the RoyalMD
✨ Click on the image to watch the video in 4K: 
[![Watch the video](https://img.youtube.com/vi/fwk0BlfbTHc/maxresdefault.jpg)](https://www.youtube.com/watch?v=fwk0BlfbTHc)

## QUICK RUN
```bash
python ./RoyalMD_beta.py ./test_systems/short_and_flexible.pdb
```

## 🔭 Overview 
**RoyalMD** is a lightweight, educational molecular dynamics pipeline built on **OpenMM**, designed to run protein simulations on portable hardware. It automates on the fly common MD setup steps: from PDB fixing to solvation, minimization, and production runs, using a single command-line interface. The current beta version allows running molecular dynamics of any molecular system composed of proteins, nucleic acids and their complexes. It automatically detects your hardware and uses GPU if available.


## 👤 Author

This script was developed and benchmarked by **Gleb Novikov**


## ✨ Features
- Automatic **PDB fixing** (missing atoms, residues, protonation via `pdbfixer`)
- Flexible **force field selection** for protein & water models (AMBER, CHARMM)
- **Solvation and ion placement** with configurable box geometry
- Energy **minimization** and **NPT production runs**
- Flexible trajectory output and logging for benchmarks
- Designed for **Mac/Linux**

---

## 🪄 Requirements
- Python 3.12
- OpenMM
- pdbfixer
- mdtraj (for saving MD trajectories in the NetCDF format)

```bash
conda create -n MDsims python=3.12
conda activate MDsims
conda install -c conda-forge openmm pdbfixer mdtraj
conda install -c conda-forge "numpy<2.0" # <<< if you have any numpy-related errors"
```

## ⚜️ Ussage

```bash
./RoyalMD_beta.py your_structure.pdb
```

## ⚙️ Configuration

Customize all variables in the MAIN function of the script.

```bash
 # --- CONFIGURATION ---
input_pdb = sys.argv[1]  # use pdb file directly from terminal
pre_sim_pdb = 'solvated.pdb'  # name of the solvated system for visualization
output_nc = 'production.nc'  # trajectory file generated during the simulation
timestep = 0.002  # integration timestep (in ps)
total_steps = 500000  # total number of steps (1 ns of production run; increase for longer simulations)
report_interval = 1500  # interval for saving snapshots in the trajectory
ligands_to_remove = ['SO4','EDO', 'LIG', 'LIH', 'lig', 'lih']  # small molecules to filter out from input PDB
temperature = 300  # temperature for Langevin dynamics (K)
pressure = 1.0  # pressure for Monte Carlo barostat (atm)
box_type = 'dodecahedron'  # type of simulation box
box_padding = 1.0  # solvent buffer around the system (nm); increase to 1.2–1.5 nm if kin. energy fluctuates
ionic_strength = 0.15  # salt concentration (M)
env_pH = 7.0  # pH for protonation during system preparation
ff_protein = 'amber14'  # force field for protein
ff_water = 'tip3p'  # force field for water
log_freq = 1  # frequency of log output during the production run

```
```

