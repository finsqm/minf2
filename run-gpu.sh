#!/bin/bash
export MINF_DATA_DIR="/home/s1220880/minf2/data"
export MINF_RESULTS_DIR="/home/s1220880/minf2/results"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/opt/cuda


cd /home/s1220880/minf2/
source .venv/bin/activate
pip install -r requirements.txt

cd src/seq2seq/wjazz

python LSTM-gpu.py
