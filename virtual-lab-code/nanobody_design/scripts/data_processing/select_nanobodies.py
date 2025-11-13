"""Select top-scoring mutated nanobodies for experimental validation."""

from pathlib import Path

import pandas as pd


def select_nanobodies(
    score_path_pattern: str,
    max_round: int,
    save_path: Path,
    top_n: int = 24,
) -> None:
    """Select top-scoring mutated nanobodies for experimental validation.

    :param score_path_pattern: The pattern to use to find score files by replacing "{round_num}" with the round number.
    :param max_round: The maximum round number to include.
    :param save_path: Path to save the selected nanobodies.
    :param top_n: Number of nanobodies to select.
    """
    # Load scores from all files
    all_scores = []

    for round_num in range(1, max_round + 1):
        # Load scores for the round
        scores = (
            pd.read_csv(score_path_pattern.format(round_num=round_num))
            .assign(round_num=round_num)
            .set_index("name")
        )

        # Correct ESM log likelihoods to compare to wildtype instead of previous mutant
        if round_num == 1:
            scores["log_likelihood_ratio_vs_wildtype"] = scores["log_likelihood_ratio"]
        else:
            previous_mutant_names = [
                "-".join(name.split("-")[:-1]) for name in scores.index
            ]

            scores["log_likelihood_ratio_vs_wildtype"] = (
                scores["log_likelihood_ratio"].to_numpy()
                + all_scores[-1]
                .loc[previous_mutant_names, "log_likelihood_ratio"]
                .to_numpy()
            )

        # Append scores
        all_scores.append(scores)

    # Combine scores from all rounds
    all_scores = pd.concat(all_scores)

    # Compute corrected weighted score
    all_scores["weighted_score_vs_wildtype"] = (
        0.2 * all_scores["log_likelihood_ratio_vs_wildtype"]
        + 0.5 * all_scores["Interface_pLDDT"]
        - 0.3 * all_scores["dG_separated"]
    )

    # Sort by weighted score
    all_scores.sort_values(
        by="weighted_score_vs_wildtype", ascending=False, inplace=True
    )

    # Select top nanobodies
    selected_nanobodies = all_scores.head(top_n)

    # Save selected nanobodies
    save_path.parent.mkdir(parents=True, exist_ok=True)
    selected_nanobodies.to_csv(save_path)


if __name__ == "__main__":
    from tap import tapify

    tapify(select_nanobodies)
