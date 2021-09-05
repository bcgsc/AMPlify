#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:15:46 2019

This script is for filtering mature AMPs to be valid AMPlify input
(Length between 2 and 200, and 20 standard aa only)

@authors: S. Austin Hammond & Chenkai Li
"""

import sys
from Bio import SeqIO

fasta = sys.argv[1]

sid = []
sqn = []
for seq_record in SeqIO.parse(fasta, 'fasta'):
    sid.append(str(seq_record.description))
    sqn.append(str(seq_record.seq))

# remove duplicates
sid_filtered = []
sqn_filtered = []+list(set(sqn))
sqn_temp = []+list(set(sqn))

l = len(sqn_temp)
aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

# only consider sequences with length [2,200] and with 20 standard aa, which are valid sequences for AMPlify
for i in range(l):
    if len(sqn_temp[i]) >200 or len(sqn_temp[i]) < 2 or set(sqn_temp[i])-set(aa) != set():
        sqn_filtered.remove(sqn_temp[i])
    else:
        sid_filtered.append(sid[sqn.index(sqn_temp[i])])

filtered = open("digested-sequences-filtered.fa", "w")
for i in range(len(sqn_filtered)):
    filtered.write('>' + sid_filtered[i] + '\n' + sqn_filtered[i] + '\n')
filtered.close()
