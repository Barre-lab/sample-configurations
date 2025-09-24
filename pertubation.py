import scm.plams
import numpy as np
from collections import deque


def get_random_atom(mol):
    """
    returns random index of an atom in the molecule
    """

    n_atoms = len(mol.atoms)
    index = np.random.randint(1, n_atoms)

    return index


def random_bond_length_scale(max_deviation):
    """
    Returns a random scale factor within the given boundary
    """
    return np.random.uniform(1 - abs(max_deviation - 1), 1 + abs(max_deviation - 1))


def random_theta(max_theta):
    """
    Returns a random angle for theta between 0 and max_theta
    """
    return np.random.uniform(0, np.deg2rad(max_theta))


def random_phi():
    """
    Returns a random value of phi between 0 and 2pi radians.
    """
    return np.random.uniform(0, 2*np.pi)


def random_rotation_angle(max_rotation):
    """
    Returns a random angle between -max_rotation and +max_rotation.
    """
    return np.random.uniform(np.deg2rad(-max_rotation), np.deg2rad(max_rotation))


def rotation_in_cone(input_v, theta, phi):
    """
    Rotates a vector around by the given spherical coordinates.

    Think of the vector being aligned with the typical z-axis with a fixed
    length which is then transformed by phi and theta. (theta is angle with
    the z-axis)

    Both angles are given in degrees.
    """

    v = np.array(input_v, float)
    L = np.linalg.norm(v)
    if L == 0:
        return v
    axis = v / L

    # local direction in spherical coordinates (cone around z-axis)
    local = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    # rotate z-axis [0,0,1] to align with axis
    zhat = np.array([0, 0, 1.0])
    v_cross = np.cross(zhat, axis)
    c = np.dot(zhat, axis)
    if np.allclose(v_cross, 0):
        rotated = local if c > 0 else -local
    else:
        s = np.linalg.norm(v_cross)
        vx = np.array([[0, -v_cross[2], v_cross[1]],
                       [v_cross[2], 0, -v_cross[0]],
                       [-v_cross[1], v_cross[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        rotated = R @ local

    return L * rotated


def perturb_bond(fragment, bond, moving_atom, bond_length_scale,
                 theta_cone_angle, phi_cone_angle, rotation_angle):
    """
    Identifies which atoms have to be translated and then applies a
    perturbation of the vector between the two provided atoms
    to all the moving atoms.

    All angles are given in degrees.

    Parameters
    ----------
    length : maximum scale to bond lengths (1.05, 1,1, 1.2 etc).
    max_angle : maximum angle of deviation from bond vector (cone shape).
    max_rotation : maximum angle of rotation around bond axis.
    """
    atoms_to_move = {moving_atom}

    def dfs(v):
        for e in v.bonds:
            if e is not bond:
                u = e.other_end(v)
                if u not in atoms_to_move:
                    atoms_to_move.add(u)
                    dfs(u)

    dfs(moving_atom)

    if len(atoms_to_move) == len(fragment):
        print("chosen bond does not divide molecule")
        return

    # Identify pivot atom (fixed) and bond vectors
    v_moving_to_pivot = np.array(bond.as_vector(start=moving_atom))
    v_pivot_to_moving = -v_moving_to_pivot
    current_length = bond.length()

    # Bond length change
    new_length = bond_length_scale * current_length

    trans_len = ((new_length / current_length) - 1.0) * v_pivot_to_moving
    v_after_length = v_pivot_to_moving + trans_len

    # Bond direction change
    v_new = rotation_in_cone(v_after_length, theta=theta_cone_angle, phi=phi_cone_angle)

    # Translation that swings the bond direction
    trans_dir = v_new - v_after_length

    # Total translation
    trans_total = trans_len + trans_dir

    # apply translation to all atoms on moving side
    xyz_array = fragment.as_array(atom_subset=atoms_to_move)
    xyz_array += trans_total
    fragment.from_array(xyz_array, atom_subset=atoms_to_move)

    if rotation_angle:
        fragment.rotate_bond(bond, moving_atom, rotation_angle)


def guarded_random_bond_pertubation(fragment, bond, moving_atom, max_bond_length_scale,
                                    max_theta_cone_angle, max_rotation_angle,
                                    max_tries=1000):
    """
    Performs perturbation of the bond but checks if this perturbation changes
    the connectivity. If this is the case, new perturbations are tried until
    the connectivity remains unchanged.

    If the maximum number of tries is exceeded, no perturbation is applied.

    All angles are given in degrees.
    """

    if not fragment.bonds:
        fragment.guess_bonds()
    ref_conn = fragment.get_connection_table()

    for _ in range(max_tries):

        # Store backup of fragment
        xyz_backup = fragment.as_array()

        # Generating random scalars and angles for perturbation
        length_scale = random_bond_length_scale(max_bond_length_scale)
        theta = random_theta(max_theta_cone_angle)
        phi = random_phi()
        rotation = random_rotation_angle(max_rotation_angle)

        # Try random attempt
        perturb_bond(fragment, bond, moving_atom,
                     bond_length_scale=length_scale,
                     theta_cone_angle=theta,
                     phi_cone_angle=phi,
                     rotation_angle=rotation)

        # Checking if connectivity has changed
        fragment.guess_bonds()
        if fragment.get_connection_table() == ref_conn:
            print("Pertubation made")
            break

        # Connectivity has changed so fragment before perturbation is restored
        fragment.from_array(xyz_backup)


def perturb_molecule_randomly(molecule, max_bond_length_scale=1.1,
                              max_theta_cone_angle=30, max_rotation_angle=30):
    """
    Picks a random atom in the molecule and applies specific perturbations
    to the bonds with the neighbors. This process is repeated such that all
    atoms and bonds are visited.

    If a bond does not divide the molecule in two parts (because both atoms
    are part of a ring) no perturbation is made to the bond. Ring structures
    are therefore kept intact.
    """

    if not isinstance(molecule, scm.plams.Molecule):
        fragment = scm.plams.Molecule(molecule)
    else:
        fragment = molecule.copy()
    fragment.guess_bonds()

    index = get_random_atom(fragment)

    # Using same index for testing
    index = 24
    initial_atom = fragment[index]
    print(f"Initial index: {index}")

    visited = {initial_atom}
    queue = deque([initial_atom])

    while queue:
        pivot = queue.popleft()

        for bond in pivot.bonds:
            neighbor = bond.other_end(pivot)
            if neighbor in visited:
                continue

            # move only the neighbor side of the bond, pivot remains fixed
            guarded_random_bond_pertubation(fragment, bond, moving_atom=neighbor,
                                            max_bond_length_scale=max_bond_length_scale,
                                            max_theta_cone_angle=max_theta_cone_angle,
                                            max_rotation_angle=max_rotation_angle)

            visited.add(neighbor)
            queue.append(neighbor)

    with open("output-mol.xyz", "w") as f:
        fragment.writexyz(f)

    return fragment
