import argparse
import numpy as np
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
from typing import List, Tuple


def parse_arguments() -> Tuple[str, int]:
    """Parse command-line arguments to get the nanobody sequence and display limit.

    Returns:
        Tuple[str, int]: The input nanobody sequence and number of top mutations to display.
    """
    parser = argparse.ArgumentParser(
        description="Identify promising point mutations in a nanobody sequence using ESM log-likelihoods."
    )
    parser.add_argument(
        "nanobody_sequence",
        type=str,
        help="The amino acid sequence of the nanobody in single-letter code.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top mutations to display (default: 10). Must be a positive integer.",
    )
    args = parser.parse_args()

    # Validate inputs
    if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in args.nanobody_sequence):
        parser.error(
            "Invalid sequence: Please ensure all characters are valid amino acid codes (A-Z)."
        )
    if args.top_n <= 0:
        parser.error("Invalid --top-n value: Must be a positive integer.")

    return args.nanobody_sequence, args.top_n


def compute_log_likelihood_ratios(
    seq: str, model, tokenizer
) -> List[Tuple[int, str, str, float]]:
    """Computes log-likelihood ratios for each possible point mutation in the sequence.

    Args:
        seq (str): The input nanobody sequence.
        model: The ESM model for masked language modeling.
        tokenizer: Tokenizer corresponding to the ESM model.

    Returns:
        List[Tuple[int, str, str, float]]: A list of tuples containing position, original amino acid, mutated amino acid, and log-likelihood ratio.
    """
    try:
        encoded_input = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        original_output = model(**encoded_input)

        log_likelihoods = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        for pos in range(1, len(seq) + 1):  # Skip [CLS] token which is at index 0
            for aa in amino_acids:
                if seq[pos - 1] == aa:
                    continue

                mutated_sequence = seq[: pos - 1] + aa + seq[pos:]
                mutated_input = tokenizer(
                    mutated_sequence, return_tensors="pt", add_special_tokens=True
                )
                mutated_output = model(**mutated_input)

                original_ll = original_output.logits[
                    0, pos, tokenizer.convert_tokens_to_ids(seq[pos - 1])
                ].item()
                mutated_ll = mutated_output.logits[
                    0, pos, tokenizer.convert_tokens_to_ids(aa)
                ].item()
                ll_ratio = mutated_ll - original_ll

                log_likelihoods.append((pos, seq[pos - 1], aa, ll_ratio))

        return sorted(log_likelihoods, key=lambda x: x[3], reverse=True)
    except Exception as e:
        print(
            f"An error occurred during computation: {e}. Please ensure your sequence is valid and model is correctly loaded."
        )
        return []


def main():
    nanobody_sequence, top_n = parse_arguments()

    try:
        print("Loading model and tokenizer...")
        model = EsmForMaskedLM.from_pretrained("facebook/esm1b-t33_650M_UR50S")
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b-t33_650M_UR50S")
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

    print("Computing log-likelihood ratios...")
    mutations = compute_log_likelihood_ratios(nanobody_sequence, model, tokenizer)

    if mutations:
        print(
            f"Top {top_n} suggested mutations (position, original_aa, mutated_aa, log_likelihood_ratio):"
        )
        for mutation in mutations[:top_n]:
            print(mutation)
    else:
        print(
            "No mutations could be computed. Please check your input sequence and model."
        )

    print("\nInterpretation:")
    print(
        "Log-likelihood ratios indicate the relative likelihood of mutations improving binding affinity."
    )
    print(
        "Higher positive values suggest potentially beneficial mutations, suitable for further experimental validation."
    )
    print(
        "Consider the biological context, such as structural data or known functional regions, when prioritizing mutations for testing."
    )


if __name__ == "__main__":
    main()
