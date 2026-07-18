#!/usr/bin/env python3
"""Patch lerobot FeatureType enum to add missing values."""
p = '/data/miniconda/envs/lingbotvla/lib/python3.12/site-packages/lerobot/configs/types.py'
with open(p) as f:
    content = f.read()

old = 'REWARD = "REWARD"'
new = 'REWARD = "REWARD"\n    LOW_DIM = "low_dim"\n    VISUAL_OLD = "visual"'

if old in content:
    content = content.replace(old, new)
    with open(p, 'w') as f:
        f.write(content)
    print('PATCHED FeatureType')
else:
    print('NOT FOUND')
    # print context
    for i, line in enumerate(content.split('\n')):
        if 'FeatureType' in line or 'REWARD' in line:
            print(f'  {i+1}: {line}')
