import os
import sys
import csv
from typing import List, Tuple
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
import argparse
import glob


def is_valid_pdb_file(pdb_file: str) -> bool:
    """
    Check if the PDB file is valid and contains expected data structure.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_file)
        return True if structure else False
    except Exception as e:
        print(f"Invalid PDB file {pdb_file}: {e}")
        return False


def calculate_interface_residues(
    structure: Chain,
    nanobody_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
) -> List[Residue]:
    """
    Identify interface residues between specified nanobody and antigen chains in a PDB structure.

    Args:
        structure (Chain): A Bio.PDB Chain object of the complex.
        nanobody_chain_id (str): Chain ID for the nanobody.
        antigen_chain_id (str): Chain ID for the antigen.
        distance_threshold (float): Distance threshold in Å to consider residues as interacting.

    Returns:
        List[Residue]: List of interface residues.
    """
    atoms = [
        atom
        for atom in structure.get_atoms()
        if atom.parent.parent.id in [nanobody_chain_id, antigen_chain_id]
    ]
    neighbor_search = NeighborSearch(atoms)

    interface_residues = set()
    for atom in atoms:
        if atom.parent.parent.id == nanobody_chain_id:
            target_chain_id = antigen_chain_id
        elif atom.parent.parent.id == antigen_chain_id:
            target_chain_id = nanobody_chain_id
        else:
            continue

        nearby_atoms = neighbor_search.search(atom.coord, distance_threshold)
        for nearby_atom in nearby_atoms:
            if nearby_atom.parent.parent.id == target_chain_id:
                interface_residues.add(atom.parent)

    return list(interface_residues)


def calculate_interface_pLDDT(
    pdb_file: str,
    nanobody_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
) -> Tuple[str, float, int, int]:
    """
    Calculate the interface pLDDT score for a given PDB file.

    Args:
        pdb_file (str): Path to the PDB file.
        nanobody_chain_id (str): Chain ID for the nanobody.
        antigen_chain_id (str): Chain ID for the antigen.
        distance_threshold (float): Distance threshold for defining interface residues.

    Returns:
        Tuple[str, float, int, int]: PDB filename, computed interface pLDDT score, number of interface residues, and number of interface atoms.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return pdb_file, 0.0, 0, 0

    if nanobody_chain_id not in [
        chain.id for chain in structure.get_chains()
    ] or antigen_chain_id not in [chain.id for chain in structure.get_chains()]:
        print(
            f"Chain IDs {nanobody_chain_id} or {antigen_chain_id} not found in {pdb_file}."
        )
        return pdb_file, 0.0, 0, 0

    interface_residues = calculate_interface_residues(
        structure, nanobody_chain_id, antigen_chain_id, distance_threshold
    )
    if not interface_residues:
        print(f"No interface residues found in {pdb_file}.")
        return pdb_file, 0.0, 0, 0

    total_plddt_score = sum(
        atom.bfactor for residue in interface_residues for atom in residue
    )
    atom_count = sum(len(residue) for residue in interface_residues)
    residue_count = len(interface_residues)

    average_plddt = total_plddt_score / atom_count if atom_count > 0 else 0.0
    return pdb_file, average_plddt, residue_count, atom_count


def process_directory(
    directory: str,
    nanobody_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
    output_file: str,
) -> None:
    """
    Process all PDB files in a directory and output the interface pLDDT scores to a CSV file.

    Args:
        directory (str): Path to the directory containing PDB files.
        nanobody_chain_id (str): Chain ID for the nanobody.
        antigen_chain_id (str): Chain ID for the antigen.
        distance_threshold (float): Distance threshold for defining interface residues.
        output_file (str): Path to the output CSV file.
    """
    pdb_files = glob.glob(
        os.path.join(directory, "**/*unrelaxed_rank_001*.pdb"), recursive=True
    )
    if not pdb_files:
        print(f"No PDB files found in the directory '{directory}'.")
        return

    results = [
        calculate_interface_pLDDT(
            pdb_file, nanobody_chain_id, antigen_chain_id, distance_threshold
        )
        for pdb_file in pdb_files
    ]

    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "PDB_File",
                "Interface_pLDDT",
                "Interface_Residue_Count",
                "Interface_Atom_Count",
            ]
        )

        for result in results:
            csv_writer.writerow(result)

    print(f"Interface pLDDT scores have been written to {output_file}")


def main():
    """
    Main function to parse command line arguments and process the PDB files.
    """
    parser = argparse.ArgumentParser(
        description="Calculate interface pLDDT scores for nanobody-antigen complexes."
    )
    parser.add_argument(
        "directory", type=str, help="Directory path containing PDB files"
    )
    parser.add_argument("nanobody_chain_id", type=str, help="Chain ID for the nanobody")
    parser.add_argument("antigen_chain_id", type=str, help="Chain ID for the antigen")
    parser.add_argument("output_file", type=str, help="Output CSV file path")
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=4.0,
        help="Distance threshold in Å for interface residues (default: 4.0)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"The provided path '{args.directory}' is not a valid directory.")
        sys.exit(1)

    process_directory(
        args.directory,
        args.nanobody_chain_id,
        args.antigen_chain_id,
        args.distance_threshold,
        args.output_file,
    )


if __name__ == "__main__":
    main()
