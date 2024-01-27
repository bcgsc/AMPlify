#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:04:20 2024

This script is for getting putative AMP sequences from AMPlify (v2.0.0) predictions

Usage: get_putative_AMPs.py [AMPlify prediction results in tsv] 

@author: Chenkai Li
"""

import sys
import pandas as pd

pred_file = sys.argv[1]

pred_results = pd.read_csv(pred_file, na_filter = False, sep = '\t')

amps = pred_results[pred_results['Prediction']=='AMP'].reset_index(drop=True)
amp_id = list(amps.Sequence_ID)
amp_seq = list(amps.Sequence)

out = open("predicted_AMPs.fasta", "w")
for i in range(len(amp_id)):
    out.write('>' + amp_id[i] + '\n' + amp_seq[i] + '\n')
out.close()