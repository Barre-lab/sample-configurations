#!/usr/bin/env amspython

import argparse
import random
import traceback
import os
import time
from typing import List
import numpy as np
import scm.plams
from scm.plams import to_smiles
from scm.plams.interfaces.molecule.rdkit import from_smiles


def parse_arguments():
    """
    Parse cli arguments to script
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "input",
            type=str,
            help="Input xyz file with fragment")

    parser.add_argument(
            "ncycles",
            type=int,
            help="The number of cycles (frames) to generate and test")

    parser.add_argument(
            "-m", "--molecules",
            type=str,
            nargs="+",
            default="O",
            required=True,
            help="The list of molecule types to generate around the fragment in smiles format")

    parser.add_argument(
            "-n", "--number-molecules",
            type=int,
            nargs="+",
            required=True,
            help="The numbers of molecules to generate for each molecule in the molecules list")

    parser.add_argument(
            "-c", "--charge",
            type=int,
            default=0,
            help="The total charge of the fragment")

    parser.add_argument(
            "-f", "--fix-atoms",
            type=int,
            nargs="*",
            help="The indices of atoms to fix in place during the geometry optimization")

    parser.add_argument(
            "-nc", "--ncores",
            type=int,
            default=8,
            help="The number of nodes to use for each calculation")

    return parser.parse_args()


def random_rotation_matrix():
    """
    Generates and returns a random rotation matrix
    """

    rng = np.random.default_rng()
    a, b, c = rng.random(3) * 2 * np.pi

    matrix = np.zeros((3, 3))

    matrix[0, 0] = np.cos(a) * np.cos(b)
    matrix[0, 1] = np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c)
    matrix[0, 2] = np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)

    matrix[1, 0] = np.sin(a) * np.cos(b)
    matrix[1, 1] = np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c)
    matrix[1, 2] = np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)

    matrix[2, 0] = -np.sin(b)
    matrix[2, 1] = np.cos(b) * np.sin(c)
    matrix[2, 2] = np.cos(b) * np.cos(c)

    return matrix


def get_total_distance(molecule):
    """
    Finds total distance of all bonded atoms in molecule
    """

    total_distance = 0.0
    for bond in molecule.bonds:
        total_distance += bond.length()
    return total_distance


def find_box(molecule, margin=2):
    """
    Finds box that fits around molecule by specified margin.

    Returned box is a matrix of following form: [min_x, max_x, min_y, max_y, min_z, max_z]
    """

    box = np.zeros(6)

    natoms = len(molecule)

    x, y, z = np.zeros(natoms), np.zeros(natoms), np.zeros(natoms)

    index = 0
    for atom in molecule:
        x[index] = atom.coords[0]
        y[index] = atom.coords[1]
        z[index] = atom.coords[2]
        index += 1

    box[0] = min(x) - margin
    box[1] = max(x) + margin
    box[2] = min(y) - margin
    box[3] = max(y) + margin
    box[4] = min(z) - margin
    box[5] = max(z) + margin

    return box


def generate_random_point(box):
    """
    Generates a random point within a given box of form: [min_x, max_x, min_y, max_y, min_z, max_z]
    """

    x = random.uniform(box[0], box[1])
    y = random.uniform(box[2], box[3])
    z = random.uniform(box[4], box[5])

    return np.array([x, y, z])


def get_configuration(input_fragment: scm.plams.Molecule, molecules: List[scm.plams.Molecule]):
    """
    Randomly rotates and positions each molecule in the molecules list around the fragment in
    an iterative way.

    The allowed distance interval between each added molecule and the fragment is specified as
    (2.5, 4) Angstrom and the minimum distance between each added molecule is 2.5 Angstrom.
    """

    # Making copy of input fragment, to not modify the input fragment itself
    fragment = input_fragment.copy()

    # Translating molecules such that their center of mass is in the origin
    for molecule in molecules:
        center_of_mass = np.array(molecule.get_center_of_mass())
        molecule.translate(-1 * center_of_mass)

    # Translating the fragment such that the center of mass is in the origin
    center_of_mass = np.array(fragment.get_center_of_mass())
    fragment.translate(-1 * center_of_mass)

    # Finding box around fragment
    box = find_box(fragment)

    # Creating a copy of the fragment without water as a future reference for distances
    fragment_no_solvent = fragment.copy()

    for mol_index, molecule in enumerate(molecules, start=1):

        # Initializing a new solvent molecule
        new_molecule = molecule.copy()
        new_molecule.rotate(random_rotation_matrix())

        # Getting the new solvent molecule into a correct position
        correct_position = False
        index = 1
        while not correct_position:

            # Translating water to new random position
            molecule_copy = new_molecule.copy()
            random_point = generate_random_point(box)
            molecule_copy.translate(random_point)

            # Evaluating distance between new water and the fragment with and without water
            distance_fragment = molecule_copy.distance_to_mol(fragment_no_solvent)
            distance_solvents = molecule_copy.distance_to_mol(fragment)
            if 2.5 < distance_fragment < 4 and distance_solvents > 2.5:
                correct_position = True

            elif index > 500:
                raise ValueError(f"Cannot add molecule {to_smiles(molecule)}, number {mol_index}")

            index += 1

        # Adding the final (correct) water molecule from the while loop
        fragment += molecule_copy

    return fragment


def calculate_energy(args, fragment):
    """
    Calculates the total energy of the given fragment after performing a geometry optimization
    """

    # Fixing provided atoms in place
    if args.fix_atoms:
        for atom in fragment:
            if fragment.index(atom) in args.fix_atoms:
                atom.properties.region = {"Fixed"}

    settings = scm.plams.Settings()
    settings.input.DFTB.Model = "GFN1-xTB"
    settings.input.ams.Task = "GeometryOptimization"
    settings.input.ams.System.Charge = args.charge

    if args.fix_atoms:
        settings.input.ams.Constraints.FixedRegion = "Fixed"

    scm.plams.config.job.runscript.nproc = args.ncores

    job = scm.plams.AMSJob()
    job.name = "geometry-optimization"
    job.molecule = fragment
    job.settings = settings
    job.run()

    return job


def main(args):
    """
    Main entry of the script
    """

    # Enforcing some requirements in the arguments
    if len(args.molecules) != len(args.number_molecules):
        raise ValueError("Number of molecule types and corresponding numbers is not the same")

    fragment = scm.plams.Molecule(args.input)
    molecule_types = [from_smiles(molecule) for molecule in args.molecules]
    molecule_numbers = args.number_molecules

    # Creating a list with the correct number of molecules of each molecule type
    molecules = []
    for index, molecule_type in enumerate(molecule_types):
        molecules += [molecule_type] * molecule_numbers[index]

    # Naming the output folder according to the number of generated water molecules
    nwater = args.number_molecules[args.molecules.index("O")]
    folder_name = f"configurations-{nwater}-H2O"

    # Creating folder in which configurations are optimized
    if os.path.isdir(folder_name):
        raise NameError(f"{folder_name} already exists!")
    os.mkdir(folder_name)
    os.chdir(folder_name)
    folder_dir = os.getcwd()

    # Already writing 'Input arguments' section to output file
    with open("results.txt", "x") as f:
        title = "\n" + " Input arguments ".center(81, "=") + "\n"
        f.write(title)
        for key in vars(args):
            keystr = f"{key}: ".ljust(25)
            f.write(f"{keystr} {vars(args)[key]}\n")
            f.flush()

    # Initializing some variables
    failed_cycles = 0
    total_energies = np.zeros(args.ncycles)
    configurations = np.empty(args.ncycles, dtype=object)

    for index in range(args.ncycles):

        real_index = index - failed_cycles
        os.chdir(folder_dir)

        os.mkdir(str(real_index + 1))
        os.chdir(str(real_index + 1))
        try:
            random.shuffle(molecules)
            configuration = get_configuration(fragment, molecules)

            scm.plams.init()
            job = calculate_energy(args, configuration)
            total_energies[real_index] = job.results.get_energy(unit="eV")
            configurations[real_index] = job.results.get_main_molecule()
            scm.plams.finish()

        except Exception:
            failed_cycles += 1
            os.chdir(folder_dir)
            os.rmdir(str(real_index + 1))
            print(traceback.format_exc())

    os.chdir(folder_dir)
    final_energy = np.min(total_energies)
    final_index = np.argmin(total_energies)

    # Writing most stable configuration to xyz file
    with open(f"output-{nwater}-H2O.xyz", "x") as f:
        configurations[final_index].writexyz(f)

    # Writing input fragment to xyz file
    with open("input.xyz", "x") as f:
        fragment.writexyz(f)

    # Writing output file
    with open("results.txt", "a") as f:

        # Configurations section
        title = "\n" + " Configurations ".center(81, "=") + "\n"
        f.write(title)
        headers = ["Dir", "Total energy [eV]"]
        spaces = [25, 10]
        for header, space in zip(headers, spaces):
            str_header = f"{header}".ljust(space)
            f.write(str_header)
        f.write("\n")
        for direc, energy in enumerate(total_energies, start=1):
            f.write(f"{direc}".ljust(24) + f"{energy}" + "\n")

        # Best configuration
        title = "\n" + " Best configuration (lowest energy) ".center(81, "=") + "\n"
        f.write(title)
        variable_names = ["Directory", "Energy [eV]"]
        variables = [final_index + 1, final_energy]
        for name, variable in zip(variable_names, variables):
            str_name = f"{name}: ".ljust(25)
            f.write(f"{str_name} {variable}\n")


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
