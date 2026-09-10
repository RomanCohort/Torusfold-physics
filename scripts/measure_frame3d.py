"""Superseded.

Its parser read only ATOM records. In 1EHZ the 14 modified nucleotides are HETATM, so
they were silently dropped, leaving numbering gaps: every local frame built across a gap
used a wrong backbone step b = P[i+1] - P[i]. The numbers this script produced were
contaminated and are not comparable with the corrected ones.

Use scripts/test_reconstruction_accuracy.py, which loads through scripts/truth_1ehz.py.
"""
print(__doc__)
