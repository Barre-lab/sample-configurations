#!/usr/bin/env amspython

import argparse
import random
import time
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


def main(args):
    """
    Main part of script
    """

    fragment = scm.plams.Molecule(args.input)
    water = scm.plams.Molecule("water.xyz")

    # Translating water such that oxygen is in the origin
    coords_oxygen = np.array(water[1].coords)
    water.translate(-1 * coords_oxygen)

    # Translating the fragment such that the center of mass is in the origin
    center_of_mass = np.array(fragment.get_center_of_mass())
    fragment.translate(-1 * center_of_mass)

    box = find_box(fragment)

    # Initializing fragment movie, showing progression of adding water
    with open("fragment.xyz", "w") as f:
        fragment.writexyz(f)

    # Creating a copy of the fragment without water as a future reference for distances
    fragment_no_water = fragment.copy()

    for nwater in range(args.nwater):

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
                raise ValueError(f"Cannot add water atom number {nwater}")

            index += 1

        # Adding the final (correct) water molecule from the while loop
        fragment += water_copy

        with open("fragment.xyz", "a") as f:
            fragment.writexyz(f)



    #with open("rotated_water.xyz", "w") as f:
    #    for frame in range(1000):
    #        water.rotate(random_rotation_matrix())
    #        water.writexyz(f)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
