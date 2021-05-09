#!/bin/bash

ngenes=250
model=densenet121
window=224
for patient in `python3 -m stnet patients`
do
    bin/cross_validate.py output/${model}_${window}/top_${ngenes}/${patient}_ 4 1 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 64 --workers 8 --gene_n ${ngenes} --norm
done