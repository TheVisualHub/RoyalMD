# 👑 Royal MD (ver 0.999 beta): run dynamics simulations on portable hardware
# - Critical FIX: added saving minimized snapshot it CIF format
# Created and developed by Gleb Novikov
# To use on Mac or Linux terminals:
# ./RoyalMD_beta.py your_structure.pdb
# (c) The VisualHub 2026
# For eductional use only

import os
import sys
import glob
import time
from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer

# --- STEP 0: CLEAN OLD DATA ---
def clean_old_files(target_pdb, activate: bool = False):
    if not activate:
        return
    print(f"🧹Cleaning Old Data:")
    patterns = ["*.dcd", "*.nc", "*cif", target_pdb]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

# --- STEP 1: FIX & CLEAN TOPOLOGY ---
def fix_topology(input_pdb, env_pH, ligands_to_remove):
    print(f"--- Step 1: Fix & Clean Topology ({input_pdb}) ---")
    fixer = PDBFixer(filename=input_pdb)

    # === NEW BLOCK: CONVERT MSE TO MET ===
    # This must run BEFORE we process atoms or missing residues
    mse_count = 0
    for residue in fixer.topology.residues():
        if residue.name == 'MSE':
            residue.name = 'MET'
            mse_count += 1
            for atom in residue.atoms():
                if atom.element.symbol == 'Se':
                    # Convert Selenium (Se) to Sulfur (S)
                    atom.element = Element.getBySymbol('S')
                    atom.name = 'SD' # Standard name for Sulfur in MET
    
    if mse_count > 0:
        print(f"🧬 Converted {mse_count} 'MSE' residues to 'MET'.")
    # =====================================

    # 1. Handle Gaps
    fixer.findMissingResidues()
    
    # 2. Blacklist Logic (Remove unwanted ligands)
    modeller = Modeller(fixer.topology, fixer.positions)
    blacklist = set(ligands_to_remove)
    
    residues_to_strip = [r for r in modeller.topology.residues() if r.name in blacklist]

    # Optional: Log kept residues
    unique_residues_kept = {r.name for r in modeller.topology.residues() if r.name not in blacklist}
    print(f"ℹ️  Residues being preserved: {sorted(list(unique_residues_kept))}")

    if residues_to_strip:
        print(f"🧹 Cleaning: Removing {len(residues_to_strip)} unwanted residues/ligands.")
        removed_names = {r.name for r in residues_to_strip}
        print(f"   (Removed types: {removed_names})")
        modeller.delete(residues_to_strip)
    else:
        print("✨ No blacklisted residues found. Keeping all original atoms.")

    # Update fixer
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    # 3. Add Hydrogens / Missing Atoms
    print(f"⚛️ Filling missing atoms and hydrogens at pH '{env_pH}'...")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(env_pH)
    return fixer

# --- STEP 2: SOLVATION ---
def solvate_system(fixer, ff_protein, ff_water, ff_map, box_type, box_padding, ionic_strength, pre_sim_pdb):
    print(f"-- Step 2: Solvation 🫧 {box_type} box with {box_padding} nm --")
    protein_xml = ff_map[ff_protein]

    try:
        water_xml = f"{ff_protein}/{ff_water}.xml"
        forcefield = ForceField(protein_xml, water_xml)
    except ValueError:
        water_xml = f"{ff_water}.xml"
        forcefield = ForceField(protein_xml, water_xml)

    print(f"🥂 Successfully loaded: {protein_xml} with {water_xml}")

    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(forcefield, 
                        padding=box_padding * nanometer, 
                        boxShape=box_type, 
                        ionicStrength=ionic_strength * molar)

    print(f"Final System Composition: {modeller.topology.getNumAtoms()} atoms.")
    with open(pre_sim_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    return modeller, forcefield

# --- STEP 3: SETUP SYSTEM & MINIMIZE ---
def setup_and_minimize(modeller, forcefield, temperature, pressure, timestep):
    print("--- Step 3: Setup System & Minimize ⚡ ---")
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
    system.addForce(MonteCarloBarostat(pressure*bar, temperature*kelvin))
    integrator = LangevinMiddleIntegrator(temperature*kelvin, 1/picosecond, timestep*picoseconds)

    print("✨ Minimizing system ... ", end="", flush=True)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    print("DONE!")

    print("👏 Minimization FINISHED!")

    # --- DUMP MINIMIZED SNAPSHOT ---
    min_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
    current_positions = min_state.getPositions()
    with open('minimized.cif', 'w') as f:
        PDBxFile.writeFile(simulation.topology, current_positions, f)
    
    return simulation, system

# --- STEP 4: PRODUCTION RUN ---
def run_production(simulation, system, total_steps, timestep, report_interval, output_nc, log_freq, temperature, pressure):
    print(f"--- Step 4: Production Run (NPT) 👑 at {temperature} K and {pressure} ATM ---")
    
    # Detect GPU
    platform = simulation.context.getPlatform()
    print(f"✨ Running on Platform: {platform.getName()}")
    if platform.getName() in ['CUDA', 'OpenCL', 'Metal']:
        device_name = platform.getPropertyValue(simulation.context, 'DeviceName')
        print(f"💠 Active GPU: {device_name}")
    else:
        print("💻 Running on CPU (no GPU detected or selected)")

    # Trajectory Reporter
    try:
        from mdtraj.reporters import NetCDFReporter
        simulation.reporters.append(NetCDFReporter(output_nc, report_interval))
    except:
        simulation.reporters.append(DCDReporter('production.dcd', report_interval))

    # Custom Output Logic
    steps_per_percent = int(total_steps / 100)
    print(f"\n🔮 Simulation Duration: {total_steps * timestep / 1000:.2f} ns")
    print(f"\n{'%':>4} {'Step':>10} {'PE':>15} {'KE':>8} {'ns/day':>10}")
    print("-" * 55)

    production_start = time.time()
    last_time = time.time()

    for i in range(1, 101):
        simulation.step(steps_per_percent)
        
        if i % log_freq == 0:
            state = simulation.context.getState(getEnergy=True)
            pot_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            kin_energy = state.getKineticEnergy().value_in_unit(kilojoules_per_mole)
            
            current_time = time.time()
            elapsed = current_time - last_time
            speed = (steps_per_percent * log_freq * timestep * 0.001) / (elapsed / 86400)
            
            print(f"{i:>3}% {simulation.currentStep:>10} {pot_energy:>15.1f} {kin_energy:>8.1f} {speed:>10.1f}")
            last_time = current_time

    # Final Summary
    production_end = time.time()
    total_seconds = production_end - production_start
    avg_speed = (total_steps * timestep * 0.001) / (total_seconds / 86400)

    print("-" * 55)
    print(f"WORK COMPLETED!")
    print(f"⏳Total wall-clock time: {int(total_seconds // 60)}m {int(total_seconds % 60)}s")
    print(f"⚜️ Average performance: {avg_speed:.2f} ns/day")
    print(f"🌀Trajectory saved to: {output_nc}")

# --- MAIN CONTROLLER ---
def main():
    if len(sys.argv) < 2:
        print("ERROR: No input PDB file provided.")
        print("Usage: python RoyalMD_beta.py your_structure.pdb")
        sys.exit(1)

    # --- CONFIGURATION ---
    input_pdb = sys.argv[1] 
    pre_sim_pdb = 'solvated.pdb'
    output_nc = 'production.nc'
    timestep = 0.002
    total_steps = 500000 
    report_interval = 1500 
    ligands_to_remove = ['SO4','EDO', 'LIG', 'LIH', 'lig', 'lih']
    temperature = 300 
    pressure = 1.0 
    box_type = 'dodecahedron'
    box_padding = 1.0
    ionic_strength = 0.15
    env_pH = 7.0
    ff_protein = 'amber14'
    ff_water = 'tip3p'
    log_freq = 1

    ff_map = {
        'amber14': 'amber14-all.xml',
        'amber99sb': 'amber99sb.xml',
        'amber99sbildn': 'amber99sbildn.xml',
        'amber03': 'amber03.xml',
        'charmm36': 'charmm36.xml'
    }

    # --- 👑 EXECUTION PIPELINE 👑 ---
    clean_old_files(pre_sim_pdb, activate=True) # Step 0
    
    fixer = fix_topology(input_pdb, env_pH, ligands_to_remove) # Step 1
    
    modeller, forcefield = solvate_system(fixer, ff_protein, ff_water, ff_map, 
                                          box_type, box_padding, ionic_strength, 
                                          pre_sim_pdb) # Step 2
    
    simulation, system = setup_and_minimize(modeller, forcefield, 
                                            temperature, pressure, timestep) # Step 3
    
    run_production(simulation, system, total_steps, timestep, 
                   report_interval, output_nc, log_freq, 
                   temperature, pressure) # Step 4

if __name__ == "__main__":
    main()