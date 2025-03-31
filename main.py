#!/usr/bin/env amspython

import argparse
import random
import traceback
import os
import numpy as np
import scm.plams


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
            "nwater",
            type=int,
            help="The number of water molecules to place around the input fragment")

    parser.add_argument(
            "ncycles",
            type=int,
            help="The number of cycles (frames) to generate and test")

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


def get_configuration(input_fragment, water, nwater):
    """
    Gets configuration of the specified number of water molecules surrounding the fragment
    """

    # Making copy of input fragment, to not modify the input fragment itself
    fragment = input_fragment.copy()

    # Translating water such that oxygen is in the origin
    coords_oxygen = np.array(water[1].coords)
    water.translate(-1 * coords_oxygen)

    # Translating the fragment such that the center of mass is in the origin
    center_of_mass = np.array(fragment.get_center_of_mass())
    fragment.translate(-1 * center_of_mass)

    # Finding box around fragment
    box = find_box(fragment)

    # Creating a copy of the fragment without water as a future reference for distances
    fragment_no_water = fragment.copy()

    for index in range(nwater):

        # Initializing a new water molecule
        new_water = water.copy()
        new_water.rotate(random_rotation_matrix())

        # Getting the new water molecule into a correct position
        correct_position_water = False
        index = 1
        while not correct_position_water:

            # Translating water to new random position
            water_copy = new_water.copy()
            random_point = generate_random_point(box)
            water_copy.translate(random_point)

            # Evaluating distance between new water and the fragment with and without water
            distance_fragment = water_copy.distance_to_mol(fragment_no_water)
            distance_water = water_copy.distance_to_mol(fragment)
            if 2.5 < distance_fragment < 4 and distance_water > 2.5:
                correct_position_water = True

            elif index > 200:
                raise ValueError(f"Cannot add water atom number {index}")

            index += 1

        # Adding the final (correct) water molecule from the while loop
        fragment += water_copy

    return fragment


def calculate_energy(args, fragment):
    """
    Calculates the total energy of the given fragment with an optional geometry optimization
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

    fragment = scm.plams.Molecule(args.input)
    water = scm.plams.Molecule("/home/barre/master_thesis/scripts/sample-configurations/water.xyz")

    folder_name = f"configurations-{args.nwater}-H2O"

    # Creating folder in which configurations are tested
    if os.path.isdir(folder_name):
        raise NameError(f"{folder_name} already exists!")
    os.mkdir(folder_name)
    os.chdir(folder_name)
    folder_dir = os.getcwd()

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
            configuration = get_configuration(fragment, water, args.nwater)

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
    with open(f"output-{args.nwater}-H2O.xyz", "x") as f:
        configurations[final_index].writexyz(f)

    # Writing input fragment to xyz file
    with open("input.xyz", "x") as f:
        fragment.writexyz(f)

    # Writing output file
    with open("results.txt", "x") as f:

        # Input arguments section
        title = "\n" + " Input arguments ".center(81, "=") + "\n"
        f.write(title)
        for key in vars(args):
            keystr = f"{key}: ".ljust(25)
            f.write(f"{keystr} {vars(args)[key]}\n")

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
