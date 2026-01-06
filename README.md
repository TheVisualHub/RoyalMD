# 👑 Welcome to the RoyalMD

**RoyalMD** is a lightweight, educational molecular dynamics pipeline built on **OpenMM**, designed to run protein simulations on portable hardware. It automates common MD setup steps—from PDB fixing to solvation, minimization, and production dynamics, using a single command-line interface. The current demo version allows running molecular dynamics of any molecular system composed of proteins and nucleic acids.

---

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
- mdtraj (for saving trajecotries in NetCDF format)

```bash
conda create -n MDsims python=3.12
conda activate MDsims
conda install -c conda-forge openmm pdbfixer mdtraj
conda install -c conda-forge "numpy<2.0" # <<< if you have any numpy-related errors"
```

## ⚜️ Ussage

```bash
./RoyalMD_demo.py your_structure.pdb
```

