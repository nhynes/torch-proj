#!/usr/bin/env python

import argparse
import glob
import os
import os.path as path

import h5py
import numpy as np
import pandas as pd

PROJ_ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_ROOT = path.join(PROJ_ROOT, 'data')

# =====================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--out', default=path.join(DATA_ROOT, 'data.h5'))
args = parser.parse_args()
# =====================================================================================

data = {}
for part in ['train', 'val', 'test']:
    data[part] = None

with h5py.File(args.out, 'w') as f_ds:
    for part,part_data in data.items():
        for col in part_data.columns:
            f_ds.create_dataset(f'/{part}/{col}', data=part_md[col].as_matrix())
