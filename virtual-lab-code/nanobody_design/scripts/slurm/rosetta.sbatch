#!/bin/bash
#
#SBATCH --job-name=rosetta
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-79
#SBATCH --partition=jamesz,owners
#SBATCH --output=/oak/stanford/groups/jamesz/swansonk/slurm/output/rosetta/%a.out
#SBATCH --error=/oak/stanford/groups/jamesz/swansonk/slurm/error/rosetta/%a.err
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=swansonk@stanford.edu

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"
echo "working directory=$SLURM_SUBMIT_DIR"

source /home/users/swansonk/.bashrc

ml biology rosetta

cd /oak/stanford/groups/jamesz/swansonk/virtual_lab || exit

experiments=()

DIR_NAME="nanobody_design/designed/workflow_1"

ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
OUTPUT_DIR="${DIR_NAME}/round_${ROUND_NUM}/rosetta/${NANOBODY}"
mkdir -p "${OUTPUT_DIR}"

for FILE in "${DIR_NAME}/round_${ROUND_NUM}/alphafold/structures/${NANOBODY}"/*/*unrelaxed_rank_001*.pdb
do
NAME=$(basename "$(dirname "$FILE")")
experiments+=("rosetta_scripts.default.linuxgccrelease \
    -s $FILE \
    -parser:protocol nanobody_design/scripts/workflow_1/models/improved/rosetta.xml \
    -out:file:scorefile ${OUTPUT_DIR}/${NAME}.sc \
    -out:path:pdb ${OUTPUT_DIR}")
done
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "Number of experiments = ${#experiments[@]}"
echo "$command"
time (eval "$command")

echo "Done"
exit 0
