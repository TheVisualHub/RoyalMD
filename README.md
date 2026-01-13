# 👑 Welcome to the RoyalMD
✨ Click on the image to watch the video in 4K: 
[![Watch the video](https://img.youtube.com/vi/fwk0BlfbTHc/maxresdefault.jpg)](https://www.youtube.com/watch?v=fwk0BlfbTHc)

## 🧬 QUICK LAUNCH
```bash
python ./RoyalMD.py ./test_systems/short_and_flexible.pdb
```

## 🔭 Overview 
**RoyalMD** is a lightweight, educational molecular dynamics pipeline built on **OpenMM**, designed to run protein simulations on portable hardware. It automates on the fly common MD setup steps: from PDB fixing to solvation, minimization, and production runs, using a single command-line interface. The current beta version allows running molecular dynamics of any molecular system composed of proteins, nucleic acids and their complexes. It automatically detects your hardware and uses GPU if available.


## 👤 Author

This script was developed and benchmarked by **Gleb Novikov**


## ✨ Features
- Automatic **PDB fixing** (missing atoms, residues, protonation via `pdbfixer`)
- Flexible **force field selection** for protein & water models (AMBER, CHARMM36)
- **Solvation and ion placement** with configurable box geometry
- Support for three-site (TIP3P) and four-site water models
- Energy **minimization** and multi-step **NPT equilibration**
- Production MD runs with benchmark-ready logging
- Cross-platform optimization with GPU detection

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

## ⚜️ Usage Examples

```bash
# Short peptide devived from 3CL protease (PDB: 6LU7):
python ./RoyalMD.py ./test_systems/short_and_flexible.pdb

# Protein-Nucleic Acid complex (RNase H, PDB: 2QKB):
python ./RoyalMD.py ./test_systems/DNA_RNA_prot.pdb

# Nucleosome (Histone octamer core wrapped by DNA, PDB: 1KX5):
python ./RoyalMD.py ./test_systems/nucleosome.pdb

# Large System (500K atoms): Immunoglobulin (PDB: 1IGT):
python ./RoyalMD.py ./test_systems/antibody.pdb
```

## ⚙️ Configuration

Customize these variables in the MAIN function of the script.

```bash
 # --- CONFIGURATION ---
 input_pdb = sys.argv[1]
    pre_sim_pdb = 'solvated.pdb'
    output_nc = 'production.nc'
    timestep = 0.002
    equil_time = 200 # 200ps of NPT equilibration (after 100ps of heating)
    production_time = 1000 # 1 ns of production
    report_interval = 1500 # report info every 1500 steps
    log_freq = 10
    # conditions
    temperature = 310 
    pressure = 1.0 
    # box and solvation: dodecahedron or octahedron
    box_type = 'dodecahedron'
    box_padding = 1.0 # if temperature fluctuates (with very flexible systems), increase it to 1.5
    ionic_strength = 0.15
    env_pH = 7.0
    # main params
    ff_protein = 'amber14' # NB: use amber14 with tip3p water
    ff_water = 'tip3p' # tip3p - with amber14/ amber99; opc - with amber19; water - with charmm36
    ligands_to_remove = ['SO4','HOH', 'EDO', 'LIH', 'LIG'] # .. add more from your PDBs

```

## 🌀 Supported force fields and water models

```bash
Customize these variables in the MAIN function of the script.
  ff_map = {
        'amber19': 'amber19-all.xml', # with OPC water
        'amber14': 'amber14-all.xml', # with TIP3P water
        'amber99sb': 'amber99sb.xml', # with TIP3P water
        'amber99sbildn': 'amber99sbildn.xml', # with TIP3P water
        'amber03': 'amber03.xml', # with TIP3P water
        'charmm36': 'charmm36.xml' # with "water" water ;-)
    }
```
