#!/usr/bin/env amspython

import numpy as np
import scm.plams
import argparse


def parse_arguments():
    """
    Parse cli arguments to script
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-input",
            type=str,
            help="Input xyz file with fragment")

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


def main(args):
    """
    Main part of script
    """

    fragment = scm.plams.Molecule(args.input)
    water = scm.plams.Molecule("water.xyz")

    # Translating the fragment such that the origin in the center of mass
    center_of_mass = fragment.get_center_of_mass()
    print(center_of_mass)
    fragment.translate(center_of_mass)

    # Translating the water such that oxygen is the origin
    x, y, z = water[1].coords
    water.translate((-x, -y, -z))

    with open("rotated_water.xyz", "w") as f:
        for frame in range(1000):
            water.rotate(random_rotation_matrix())
            water.writexyz(f)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
