#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import os.path as path

from lxml import etree
import h5py
import numpy as np

PROJ_ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_ROOT = path.join(PROJ_ROOT, 'data')

# =====================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--voc', default=path.join(DATA_ROOT, 'voc2012'))
parser.add_argument('--out-suffix', default='')
args = parser.parse_args()
# =====================================================================================

main = path.join(args.voc, 'ImageSets', 'Main')

with open(path.join(main, 'val.txt')) as f_val_ids:
    val_ids = {l.rstrip() for l in f_val_ids}

label_ids = {}
for fname in sorted(os.listdir(main)):
    split_fname = fname.split('_')
    if len(split_fname) < 2 or split_fname[0] in label_ids:
        continue
    label_ids[split_fname[0]] = len(label_ids)+1
assert len(label_ids) == 20
print(label_ids)

part_dlists = ['img_ids', 'ibps', 'n_objs', 'labels',
               'box_xs', 'box_ys', 'box_ws', 'box_hs']
for part_dlist in part_dlists:
    locals()[part_dlist] = { 'train': [], 'val': [] }

path_annos = path.join(args.voc, 'Annotations')
for path_anno in os.listdir(path_annos):
    with open(path.join(path_annos, path_anno)) as f_anno:
        anno = etree.parse(f_anno)
        imid = path.splitext(anno.findtext('filename'))[0]
        part = 'val' if imid in val_ids else 'train'
        img_ids[part].append(imid)

        objs = anno.findall('object')

        part_ibps = ibps[part]
        if len(part_ibps) == 0:
            part_ibps.append(1) # 1 for lua
        else:
            ibps[part].append(ibps[part][-1] + n_objs[part][-1])
        n_objs[part].append(len(objs))

        for obj in objs:
            labels[part].append(label_ids[obj.findtext('name')])
            box = obj.find('bndbox')
            for c,d in zip(['x', 'y'], ['w', 'h']):
                coords = [float(box.findtext(c + e)) for e in ['min', 'max']]
                mp = sum(coords)/2
                locals()['box_%ss' % c][part].append(int(mp))
                locals()['box_%ss' % d][part].append(int(coords[1] - mp))

suff = '' + ('_' + args.out_suffix if args.out_suffix else '')
with h5py.File(path.join(DATA_ROOT, 'dataset%s.h5' % suff), 'w') as f_ds:
    for part_dlist in part_dlists:
        name_comps = part_dlist.split('_')
        name = name_comps[0] + ''.join(map(str.capitalize, name_comps[1:]))
        for part, dlist in locals()[part_dlist].items():
            data = np.array(dlist) # or fill a matrix
            f_ds.create_dataset('/%s/%s' % (part, name), data=data)

    f_ds['labels'] = [l for l, i in sorted(label_ids.items(), key=lambda x: x[1])]
