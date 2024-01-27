#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:41:55 2024

This script is to filter for novel putative AMP sequences identified by AMPlify (v2.0.0)

Usage: get_novel_putative_AMPs.py [AMPlify-predicted AMPs in fasta] [known AMP sequences in fasta]
        [UniProtKB/Swiss-Prot entries annotated with AMP-related keywords in fasta]

@author: Chenkai Li
"""

import sys
from Bio import SeqIO

putative_amp_dir = sys.argv[1]
known_amp_dir = sys.argv[2]
annotated_amp_dir = sys.argv[3]

def read_fasta(fasta):
    sid = []
    sqn = []
    for seq_record in SeqIO.parse(fasta, 'fasta'):
        sid.append(str(seq_record.description))
        sqn.append(str(seq_record.seq))
    return sid, sqn

putative_amp_id, putative_amp_seq = read_fasta(putative_amp_dir)
known_amp_id, known_amp_seq = read_fasta(known_amp_dir)
annotated_amp_id, annotated_amp_seq = read_fasta(annotated_amp_dir)

for i in range(len(annotated_amp_id)):
    annotated_amp_id[i] = annotated_amp_id[i].split('|')[1]

novel_putative_amp_id = []
novel_putative_amp_seq = []

for i in range(len(putative_amp_id)):
    if putative_amp_seq[i] not in known_amp_seq and putative_amp_id[i].split('_')[1] not in annotated_amp_id:
        novel_putative_amp_id.append(putative_amp_id[i])
        novel_putative_amp_seq.append(putative_amp_seq[i])
        
out = open("novel_putative_AMPs.fasta", "w")
for i in range(len(novel_putative_amp_id)):
    out.write('>' + novel_putative_amp_id[i] + '\n' + novel_putative_amp_seq[i] + '\n')
out.close()
        