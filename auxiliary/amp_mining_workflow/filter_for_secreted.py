#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:21:41 2024

This script is to filter for candidate mature sequences with signal peptides
predicted in their corresponding parent sequences from the rAMPage cleavage results

Usage: filter_for_secreted.py [cleaved mature sequences in fasta] [ProP cleavage information in tsv]

@author: Chenkai Li
"""

import sys
import pandas as pd
from Bio import SeqIO

fasta = sys.argv[1]
cleavage = pd.read_csv(sys.argv[2], sep = '\t')

cleavage_with_signal = cleavage[cleavage['Signal Peptide']!=0].reset_index(drop=True)
cleavage_with_signal_id = list(cleavage_with_signal['Sequence'])

sid = []
sqn = []
for seq_record in SeqIO.parse(fasta, 'fasta'):
    sid.append(str(seq_record.description))
    sqn.append(str(seq_record.seq))
    
sid_filtered = []
sqn_filtered = []
for i in range(len(sid)):
    if sid[i].split('-')[0] in cleavage_with_signal_id:
        sid_filtered.append(sid[i])
        sqn_filtered.append(sqn[i])
        
filtered = open("candidate_mature_with_signal.fasta", "w")
for i in range(len(sqn_filtered)):
    filtered.write('>' + sid_filtered[i] + '\n' + sqn_filtered[i] + '\n')
filtered.close()