## Nanobody design

Starting with four nanobodies designed to bind to the original Wuhan strain of SARS-CoV-2---Ty1, H11-D4, Nb21, and VHH-72---improve them by iteratively adding mutations and scoring them using ESM, AlphaFold-Multimer, and Rosetta for four rounds.

### ESM

For round 0, copy the original nanobodies and assign an ESM log-likelihood ratio of 0.

```bash
for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python -c "from pathlib import Path; import pandas as pd
data = pd.read_csv('nanobody_design/sequences/nanobodies.csv')
data = data[data['name'] == '${NANOBODY}']
data['log_likelihood_ratio'] = 0.0
save_dir = Path('nanobody_design/designed/round_0/esm/${NANOBODY}')
save_dir.mkdir(parents=True, exist_ok=True)
data.to_csv(save_dir / '${NANOBODY}.csv', index=False)"
done
```

For all subsequent rounds, run the ESM model.

```bash
ROUND_NUM=1
PREV_ROUND_NUM=$((ROUND_NUM - 1))

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
mkdir -p nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY}
python nanobody_design/scripts/models/improved/esm.py \
    nanobody_design/designed/round_${PREV_ROUND_NUM}/scores/${NANOBODY}.csv \
    nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY}
done
```

Then, merge the ESM scores into a single file.

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python -c "from pathlib import Path;import pandas as pd
data = pd.concat([pd.read_csv(path).assign(name=path.stem) for path in Path('nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY}').glob('*.csv')])
data.to_csv('nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY}.csv', index=False)"
done
```


### ESM to AlphaFold-Multimer

Convert the top ESM-designed mutated nanobody sequences to AlphaFold-Multimer nanobody-spike sequence inputs. (For round 0, add `--nanobody_sequence_col sequence`.)

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python nanobody_design/scripts/data_processing/esm_to_alphafold.py \
    --spike_sequences_path nanobody_design/sequences/spike.csv \
    --spike_name KP.3 \
    --nanobody_sequences_dir nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY} \
    --save_dir nanobody_design/designed/round_${ROUND_NUM}/alphafold/sequences/${NANOBODY} \
    --top_n 20
done
```


### AlphaFold-Multimer

Run AlphaFold-Multimer on the nanobody-spike complexes.

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
for FILE in nanobody_design/designed/round_${ROUND_NUM}/alphafold/sequences/${NANOBODY}/*.fasta
do
NAME=$(basename "$FILE" .fasta)
colabfold_batch $FILE nanobody_design/designed/round_${ROUND_NUM}/alphafold/structures/${NANOBODY}/$NAME
done
done
```

Process the AlphaFold-Multimer complexes to extract interface pLDDT scores.

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python nanobody_design/scripts/models/improved/alphafold.py \
    nanobody_design/designed/round_${ROUND_NUM}/alphafold/structures/${NANOBODY} \
    A \
    B \
    nanobody_design/designed/round_${ROUND_NUM}/alphafold/${NANOBODY}.csv
done
```


### Rosetta

Run Rosetta to calculate interface binding energies.

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
OUTPUT_DIR="nanobody_design/designed/round_${ROUND_NUM}/rosetta/${NANOBODY}"
mkdir -p "${OUTPUT_DIR}"

for FILE in nanobody_design/designed/round_${ROUND_NUM}/alphafold/structures/${NANOBODY}/*/*unrelaxed_rank_001*.pdb
do
NAME=$(basename "$(dirname "$FILE")")
rosetta_scripts.default.linuxgccrelease \
    -s $FILE \
    -parser:protocol nanobody_design/scripts/models/improved/rosetta.xml \
    -out:file:scorefile ${OUTPUT_DIR}/${NAME}.sc \
    -out:path:pdb ${OUTPUT_DIR}
done
done
```

Then, collate the outputs.

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python nanobody_design/scripts/models/improved/rosetta.py \
    nanobody_design/designed/round_${ROUND_NUM}/rosetta/${NANOBODY} \
    nanobody_design/designed/round_${ROUND_NUM}/rosetta/${NANOBODY}.csv
done
```


### Combine scores

Combine scores from ESM, AlphaFold-Multimer, and Rosetta. (For round 0, add `--starting_sequence`.)

```bash
ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python nanobody_design/scripts/data_processing/combine_scores.py \
    --esm_scores_path nanobody_design/designed/round_${ROUND_NUM}/esm/${NANOBODY}.csv \
    --alphafold_scores_path nanobody_design/designed/round_${ROUND_NUM}/alphafold/${NANOBODY}.csv \
    --rosetta_scores_path nanobody_design/designed/round_${ROUND_NUM}/rosetta/${NANOBODY}.csv \
    --all_save_path nanobody_design/designed/round_${ROUND_NUM}/scores/${NANOBODY}_all.csv \
    --top_save_path nanobody_design/designed/round_${ROUND_NUM}/scores/${NANOBODY}.csv \
    --top_n 5
done
```


### Select nanobodies

After all the rounds have been run, select the best nanobodies.

```bash
for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python nanobody_design/scripts/data_processing/select_nanobodies.py \
    --score_path_pattern nanobody_design/designed/round_{round_num}/scores/${NANOBODY}_all.csv \
    --max_round 4 \
    --save_path nanobody_design/designed/selected/${NANOBODY}.csv \
    --top_n 24
done
```


### Prepare nanbodies for synthesis

Replace the 24th nanobody with the wildtype sequence as a control. Then, prepare the nanobodies for synthesis by adding the N-terminal pelB leader sequence (periplasm targeting) `MKYLLPTAAAGLLLLAAQPAMA` and the C-terminal His-tag with stop codon `HHHHHH*`.

```bash
mkdir -p nanobody_design/designed/combined

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
python -c "import pandas as pd
wildtype = pd.read_csv('nanobody_design/designed/round_0/scores/${NANOBODY}.csv')
wildtype['round_num'] = 0
wildtype['log_likelihood_ratio_vs_wildtype'] = 0.0
wildtype['weighted_score_vs_wildtype'] = wildtype['weighted_score']
selected = pd.read_csv('nanobody_design/designed/selected/${NANOBODY}.csv')
combined = pd.concat([selected.iloc[:23], wildtype])
combined['sequence_with_tags'] = [f'MKYLLPTAAAGLLLLAAQPAMA{sequence}HHHHHH*' for sequence in combined['sequence']]
combined.to_csv('nanobody_design/designed/combined/${NANOBODY}.csv', index=False)"
done
```


### Visualize nanobodies

Visualize the selected nanobodies using PyMOL. Below is an example for H11-D4.  For AlphaFold, replace `<color>` with `spectrum b` to show pLDDT values. For Rosetta, replace `<color>` with `util.cbc` to color by chain. For `<Draw/Ray>`, export an image using the `Draw/Ray` button in PyMOL.

Wildtype
```
load H11-D4_KP3.pdb
<color>

<Draw/Ray>

select interface, byres (chain A within 4.0 of chain B) or byres (chain B within 4.0 of chain A)
show sticks, interface
dist hbonds, (chain A), (chain B), mode=2

<Draw/Ray>
```

Mutant
```
load H11-D4-A14P-Y88V-K74T-R27L_KP3.pdb
<color>
select mut_res, chain A and resi 14+88+74+27
show sticks, mut_res
color pink, mut_res
select mut_atom, chain A and resi 14+88+74+27 and name CA
label mut_atom, resn + resi

<Draw/Ray>

select interface, byres (chain A within 4.0 of chain B) or byres (chain B within 4.0 of chain A)
show sticks, interface
dist hbonds, (chain A), (chain B), mode=2
```
