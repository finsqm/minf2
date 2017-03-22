#!/bin/bash
export MINF_DATA_DIR="/home/s1220880/minf2/data"
export MINF_RESULTS_DIR="/home/s1220880/minf2/results"
cd /home/s1220880/minf2/
source .venv/bin/activate
pip install -r requirements.txt

cd src/seq2seq/kpcorpus

python run-layers.py
