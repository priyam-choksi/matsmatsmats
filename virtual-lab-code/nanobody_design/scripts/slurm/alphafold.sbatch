#!/bin/bash
#
#SBATCH --job-name=alphafold
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-79%15
#SBATCH --partition=jamesz,owners
#SBATCH --output=/oak/stanford/groups/jamesz/swansonk/slurm/output/alphafold/%a.out
#SBATCH --error=/oak/stanford/groups/jamesz/swansonk/slurm/error/alphafold/%a.err
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

conda activate localcolabfold

cd /oak/stanford/groups/jamesz/swansonk/virtual_lab || exit

experiments=()

DIR_NAME="nanobody_design/designed/workflow_1"

ROUND_NUM=1

for NANOBODY in Ty1 H11-D4 Nb21 VHH-72
do
for FILE in "${DIR_NAME}/round_${ROUND_NUM}/alphafold/sequences/${NANOBODY}"/*.fasta
do
NAME=$(basename "$FILE" .fasta)
experiments+=("colabfold_batch $FILE ${DIR_NAME}/round_${ROUND_NUM}/alphafold/structures/${NANOBODY}/$NAME")
done
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "Number of experiments = ${#experiments[@]}"
echo "$command"
time (eval "$command")

echo "Done"
exit 0
