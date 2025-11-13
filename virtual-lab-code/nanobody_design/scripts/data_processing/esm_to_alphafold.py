"""Converts ESM mutated nanobody sequences to spike-nanobody sequences for input to AlphaFold."""

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def esm_to_alphafold(
    spike_sequences_path: Path,
    spike_name: str,
    nanobody_sequences_dir: Path,
    save_dir: Path,
    top_n: int = 10,
    spike_sequence_col: str = "rbd",
    spike_name_col: str = "name",
    nanobody_sequence_col: str = "mutated_sequence",
    nanobody_pos_col: str = "position",
    nanobody_original_aa_col: str = "original_aa",
    nanobody_mutated_aa_col: str = "mutated_aa",
    nanobody_llr_col: str = "log_likelihood_ratio",
) -> None:
    """Converts ESM mutated nanobody sequences to spike-nanobody sequences for input to AlphaFold.

    :param spike_sequences_path: Path to a CSV file containing spike protein sequences.
    :param spike_name: Name of the spike protein.
    :param nanobody_sequences_dir: Path to a directory of CSV files containing mutated nanobody sequences from ESM.
    :param save_dir: Directory to save the paired nanobody-spike sequences for input to AlphaFold.
    :param top_n: Number of top mutations to display.
    :param spike_sequence_col: Column name for the spike protein sequence in the CSV file.
    :param spike_name_col: Column name for the spike protein name in the CSV file.
    :param nanobody_sequence_col: Column name for the nanobody sequence in the CSV file.
    :param nanobody_pos_col: Column name for the nanobody mutation position in the CSV file.
    :param nanobody_original_aa_col: Column name for the original amino acid in the nanobody mutation in the CSV file.
    :param nanobody_mutated_aa_col: Column name for the mutant amino acid in the nanobody mutation in the CSV file.
    :param nanobody_llr_col: Column name for the log-likelihood ratio in the nanobody mutation in the CSV file.
    """
    # Load spike sequences
    spike = pd.read_csv(spike_sequences_path)

    # Get spike sequence and name
    spike = spike[spike[spike_name_col] == spike_name]

    if len(spike) != 1:
        raise ValueError(f"Invalid spike name '{spike_name}'.")

    spike_seq = spike[spike_sequence_col].iloc[0]
    spike_name = spike[spike_name_col].iloc[0]

    # Iterate over nanobody sequence paths
    for nanobody_sequences_path in nanobody_sequences_dir.glob("*.csv"):
        # Get nanobody sequences
        nanobody = pd.read_csv(nanobody_sequences_path)

        # Limit the number of top mutations to display
        nanobody = nanobody.sort_values(by=nanobody_llr_col, ascending=False).head(
            top_n
        )

        # Get nanobody name
        nanobody_name = nanobody_sequences_path.stem

        # Make save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Pair spike-nanobody sequences
        for nanobody_index in range(len(nanobody)):
            # Get nanobody sequence and name
            nanobody_seq = nanobody[nanobody_sequence_col].iloc[nanobody_index]

            if nanobody_sequence_col == "mutated_sequence":
                nanobody_position = nanobody[nanobody_pos_col].iloc[nanobody_index]
                nanobody_original_aa = nanobody[nanobody_original_aa_col].iloc[
                    nanobody_index
                ]
                nanobody_mutant_aa = nanobody[nanobody_mutated_aa_col].iloc[
                    nanobody_index
                ]
                nanobody_new_name = f"{nanobody_name}-{nanobody_original_aa}{nanobody_position}{nanobody_mutant_aa}"
            else:
                nanobody_new_name = nanobody_name

            complex_name = f"{nanobody_new_name}_{spike_name}"
            merged_record = SeqRecord(
                Seq(f"{nanobody_seq}:{spike_seq}"), id=complex_name, description=""
            )

            # Write the merged sequence to a new FASTA file
            with open(save_dir / f"{complex_name}.fasta", "w") as output_handle:
                SeqIO.write(merged_record, output_handle, "fasta")


if __name__ == "__main__":
    from tap import tapify

    tapify(esm_to_alphafold)
