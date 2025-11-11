import os
import pandas as pd
import argparse
import logging


def extract_scores_from_file(score_file: str) -> float:
    """
    Extract the binding energy score from a Rosetta score file.

    Parameters:
    score_file (str): Path to the score file.

    Returns:
    float: The extracted energy score.
    """
    try:
        with open(score_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("interface_energy"):
                    return float(line.split()[1])
        raise ValueError(f"No valid score found in {score_file}")
    except Exception as e:
        logging.error(f"Error processing file {score_file}: {e}")
        return None


def main(input_dir: str, output_csv: str) -> None:
    """
    Process multiple Rosetta score files and output a CSV with scores.

    Parameters:
    input_dir (str): Directory containing the score files.
    output_csv (str): Output CSV file path.
    """
    logging.basicConfig(level=logging.INFO)
    score_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".sc")
    ]
    scores = []
    errors = []

    for score_file in score_files:
        score = extract_scores_from_file(score_file)
        if score is not None:
            scores.append((os.path.basename(score_file), score))
        else:
            errors.append(os.path.basename(score_file))

    # Sort scores in ascending order assuming lower scores indicate better binding
    scores.sort(key=lambda x: x[1])

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(scores, columns=["File Name", "Score"])
    df.to_csv(output_csv, index=False)

    # Log errors if any
    if errors:
        logging.warning(
            "Encountered errors in the following files; these files may have invalid or missing data:"
        )
        for error_file in errors:
            logging.warning(f" - {error_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Rosetta score files and output CSV."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing Rosetta score files."
    )
    parser.add_argument("output_csv", type=str, help="Output CSV file path.")
    args = parser.parse_args()

    main(args.input_dir, args.output_csv)
