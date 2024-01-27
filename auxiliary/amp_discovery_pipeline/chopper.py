#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 27 2017

This script is for deriving mature AMPs from their precursors.

Read a fasta file and cut the sequences at their KR boundary.
If more than one KR, or no KR, then write sequence to log.

@authors: S. Austin Hammond & Chenkai Li
"""

import sys
import re
from Bio import SeqIO

fasta = sys.argv[1]

skipped = open("undigested-sequences.txt", "w")
chopped = open("digested-sequences.fa", "w")

sid = []
sqn = []
for seq_record in SeqIO.parse(fasta, 'fasta'):
    sid.append(str(seq_record.description))
    sqn.append(str(seq_record.seq))

for i in range(len(sqn)):
    # trypsin cuts after K or R except when followed by P
    # however, it seems that the KR is considered a
    # "typical prohormone processing signal" which
    # is between the acidic propiece and mature peptide
    # i.e. NOT part of the mature peptide, so cut it off
    kr = re.findall("KR",sqn[i])
    if kr:
        if len(kr) > 1:
            skipped.write("Sequence " + sid[i] + " not digested due to too many KR motifs\n")
            continue
        # digest the sequence, cleave between sole KR motif
        dig = sqn[i].split("KR")[-1]
        chopped.write(">" + sid[i] + "_trypsinized\n")
        chopped.write(dig + "\n")

    else:
        skipped.write("Sequence " + sid[i] + " had no KR motif and was not digested\n")

skipped.close()
chopped.close()
