#!/bin/bash
srun --ntasks=1 --gpus-per-node=1 --cpus-per-gpu=4 --mem=32G python mainfile.py 
