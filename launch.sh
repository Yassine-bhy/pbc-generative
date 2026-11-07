#!/bin/bash
# Tonight: everything the protocol allows before the pilot amendment.
#   GPU 0: baseline arm, replicates 0-4  (45 classifier runs)
#   GPU 1: cGAN pilot, then cVAE pilot, then baseline replicates 5-9 (45 runs)
# Each line is independent (";"): a crash in one pilot does not stop the rest.
# Relaunching this script after a crash continues where it stopped.
# Follow progress:  tail -f logs/gpu0.txt logs/gpu1.txt

set -u
cd "$(dirname "$0")"
mkdir -p logs

if [ ! -f models/classifier/input128_nall_seed0/best_checkpoint.pt ]; then
    echo "judge missing: models/classifier/input128_nall_seed0/best_checkpoint.pt" && exit 1
fi

nohup python train_classifier.py baseline --device cuda:0 --replicates 0 1 2 3 4 \
    > logs/gpu0.txt 2>&1 &

nohup bash -c "python train_generator.py cgan --pilot --device cuda:1; \
               python train_generator.py cvae --pilot --device cuda:1; \
               python train_classifier.py baseline --device cuda:1 --replicates 5 6 7 8 9" \
    > logs/gpu1.txt 2>&1 &

echo "launched. tail -f logs/gpu0.txt logs/gpu1.txt"
