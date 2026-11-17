#!/bin/bash
#SBATCH --job-name=bpr-corpus-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu04,gpu05
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/gxb18167/Authentic_Query_BPR/findings/external_corpus_analysis-%j.out
#SBATCH --error=/users/gxb18167/Authentic_Query_BPR/findings/external_corpus_analysis-%j.err
set -euo pipefail
cd /users/gxb18167/Authentic_Query_BPR
~/envs/bpr_fm/bin/python findings/analyze_external_corpora.py