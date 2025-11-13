import argparse
import numpy as np
import pandas as pd
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from tqdm import tqdm
import os


class NanobodyDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def parse_arguments() -> Tuple[str, str, int]:
    """Parse command-line arguments to get the CSV file path, save directory, and display limit.

    Returns:
        Tuple[str, str, int]: The input CSV file path, save directory, and number of top mutations to display.
    """
    parser = argparse.ArgumentParser(
        description="Identify promising point mutations in nanobody sequences using ESM log-likelihoods."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help='Path to the CSV file containing nanobody sequences with columns "sequence" and "name".',
    )
    parser.add_argument(
        "save_directory", type=str, help="Directory to save the output CSV files."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top mutations to display (default: 10). Must be a positive integer.",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.csv_file):
        parser.error("Invalid CSV file path: File does not exist.")
    if not os.path.isdir(args.save_directory):
        parser.error("Invalid save directory: Directory does not exist.")
    if args.top_n <= 0:
        parser.error("Invalid --top-n value: Must be a positive integer.")

    return args.csv_file, args.save_directory, args.top_n


def compute_log_likelihood_ratios(
    seq: str, model, tokenizer, batch_size: int = 16
) -> List[Tuple[int, str, str, float]]:
    """Computes log-likelihood ratios for each possible point mutation in the sequence.

    Args:
        seq (str): The input nanobody sequence.
        model: The ESM model for masked language modeling.
        tokenizer: Tokenizer corresponding to the ESM model.
        batch_size (int): Number of sequences to process in a batch.

    Returns:
        List[Tuple[int, str, str, float]]: A list of tuples containing position, original amino acid, mutated amino acid, and log-likelihood ratio.
    """
    try:
        encoded_input = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(
            "cuda"
        )
        with torch.no_grad():
            original_output = model(**encoded_input)

        log_likelihoods = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        seq_length = len(seq)

        # Prepare batches
        mutations = []
        for pos in range(1, seq_length + 1):  # Skip [CLS] token which is at index 0
            for aa in amino_acids:
                if seq[pos - 1] == aa:
                    continue
                mutated_sequence = seq[: pos - 1] + aa + seq[pos:]
                mutations.append((pos, seq[pos - 1], aa, mutated_sequence))

        # Process in batches
        for i in tqdm(
            range(0, len(mutations), batch_size), desc="Processing mutations"
        ):
            batch = mutations[i : i + batch_size]
            mutated_sequences = [mut[3] for mut in batch]
            mutated_inputs = tokenizer(
                mutated_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to("cuda")
            with torch.no_grad():
                mutated_outputs = model(**mutated_inputs)

            for j, (pos, original_aa, mutated_aa, _) in enumerate(batch):
                original_ll = original_output.logits[
                    0, pos, tokenizer.convert_tokens_to_ids(seq[pos - 1])
                ].item()
                mutated_ll = mutated_outputs.logits[
                    j, pos, tokenizer.convert_tokens_to_ids(mutated_aa)
                ].item()
                ll_ratio = mutated_ll - original_ll
                log_likelihoods.append((pos, original_aa, mutated_aa, ll_ratio))

        return sorted(log_likelihoods, key=lambda x: x[3], reverse=True)
    except Exception as e:
        print(
            f"An error occurred during computation: {e}. Please ensure your sequence is valid and model is correctly loaded."
        )
        return []


def main():
    csv_file, save_directory, top_n = parse_arguments()

    try:
        print("Loading model and tokenizer...")
        model = EsmForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(
            "cuda"
        )
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    except Exception as e:
        print(
            f"Error loading model or tokenizer: {e}. Ensure you have installed 'transformers' and 'torch'."
        )
        print("Installation steps: pip install transformers torch")
        return

    if not torch.cuda.is_available():
        print(
            "Warning: CUDA is not available. Running on CPU may be slow. Consider using a cloud service with GPU support."
        )
        print("For CUDA installation, visit: https://pytorch.org/get-started/locally/")

    # Load sequences from CSV
    df = pd.read_csv(csv_file)
    sequences = df["sequence"].tolist()
    names = df["name"].tolist()

    for seq, name in zip(sequences, names):
        print(f"Computing log-likelihood ratios for {name}...")
        mutations = compute_log_likelihood_ratios(seq, model, tokenizer)

        if mutations:
            output_file = os.path.join(save_directory, f"{name}.csv")
            with open(output_file, "w") as f:
                f.write(
                    "mutated_sequence,position,original_aa,mutated_aa,log_likelihood_ratio\n"
                )
                for mutation in mutations:
                    pos, original_aa, mutated_aa, ll_ratio = mutation
                    mutated_sequence = seq[: pos - 1] + mutated_aa + seq[pos:]
                    f.write(
                        f"{mutated_sequence},{pos},{original_aa},{mutated_aa},{ll_ratio}\n"
                    )
            print(f"Results saved to {output_file}")
        else:
            print(
                f"No mutations could be computed for {name}. Please check your input sequence and model."
            )


if __name__ == "__main__":
    main()
