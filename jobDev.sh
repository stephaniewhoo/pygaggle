#!/bin/bash
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=2 
#SBATCH --time=60:0:0 
#SBATCH --gres=gpu:v100l:2
#SBATCH --output=LTR.out

export CUDA_AVAILABLE_DEVICES=0,1
source ~/ENV/bin/activate
module load java
python -um pygaggle.run.evaluate_passage_ranker \
--split dev --method seq_class_transformer \
--model castorini/monobert-large-msmarco \
--dataset data/msmarco_ans_entire/ \
--index-dir indexes/index-msmarco-passage-20191117-0ed488 \
--task msmarco \
--output-file runs/run.monobert.ans_entire.dev.trec
