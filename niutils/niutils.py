#!/usr/bin/env python3

import os

import nibabel as nib
import numpy as np

SUB_LIST = ['001', '002', '003', '004', '007', '008', '009']
LAST_SES = 10  # 10

LAST_SES += 1

SET_DPI = 300
FIGSIZE = (18, 10)

COLOURS = ['#1f77b4ff', '#2ca02cff', '#d62728ff', '#ff7f0eff', '#ff33ccff']


#########
# Utils #
#########

def load_nifti_get_mask(fname, dim=4):
    """
    Load a nifti file and returns its data, its image, and a 3d mask.
    """
    if fname.endswith('.nii.gz'):
        fname = fname[:-7]
    img = nib.load(f'{fname}.nii.gz')
    data = img.get_fdata()
    if len(data.shape) > dim:
        for ax in range(dim, len(data.shape)):
            data = np.delete(data, np.s_[1:], axis=ax)
    data = np.squeeze(data)
    if len(data.shape) >= 4:
        mask = np.squeeze(np.any(data, axis=-1))
    else:
        mask = (data < 0) + (data > 0)
    return data, mask, img


def compute_rank(data):
    """
    Compute the ranks in the last axis of a matrix.

    It assumes that the "target" data is appended as last element in the axis.
    This is useful to compare e.g. a bunch of surrogates to real data.
    """
    reord = np.argsort(data, axis=-1)
    rank = reord.argmax(axis=-1)
    return rank/(data.shape[-1]-1)*100


def export_nifti(data, img, fname):
    """
    Export a nifti file.
    """
    out_img = nib.Nifti1Image(data, img.affine, img.header)
    if fname.endswith('.nii.gz'):
        fname = fname[:-7]
    out_img.to_filename(f'{fname}.nii.gz')


#############
# Workflows #
#############


def variance_weighted_average(fname,
                              fdir='',
                              exname='',
                              sub_list=SUB_LIST,
                              last_ses=LAST_SES):
    """
    Compute the variance weighted average of a multi-session study.

    It's supposed that:
    - all files are in the same folder
    - `fname` contains placeholders `{sub}` and `{ses}`
    """
    # Prepare dictionaries
    mask = dict.fromkeys(sub_list)
    data = dict.fromkeys(sub_list)
    data['avg'] = dict.fromkeys(sub_list)
    data['var'] = dict.fromkeys(sub_list)

    if fdir:
        fname = os.path.join(fdir, fname)
    elif os.path.split(fname)[0]:
        fdir = os.path.split(fname)[0]

    # Load niftis of all subjects
    for sub in sub_list:
        data[sub] = {}
        mask[sub] = {}
        for ses in range(1, last_ses):
            # Load data
            fname = fname.format(sub=sub, ses=f'{ses:02g}')
            data[sub][ses], mask[sub][ses], img = load_nifti_get_mask(fname, dim=3)

        # Stack in 4d (axis 3) and mask data (invert nimg mask for masked array)
        mask[sub]['stack'] = np.stack(mask[sub].values(), axis=3)
        data[sub]['stack'] = np.ma.array(np.stack(data[sub].values(), axis=3),
                                         mask=abs(mask[sub]['stack']-1))

        # Compute average & variance of masked voxels across d4
        data['avg'][sub] = data[sub]['stack'].mean(axis=3)
        data['var'][sub] = ((data[sub]['stack'] -
                             data['avg'][sub][:, :, :, np.newaxis])**2).mean(axis=3)

    # Stack subjects in 4d
    for val in ['avg', 'var']:
        data[val]['all'] = np.stack(data[val].values(), axis=3)

    # Invert variance & set infinites to zero (if any)
    invvar = 1 / data['var']['all']
    invvar[np.isinf(invvar)] = 0

    # Mask group average using invvar
    data['avg']['all'] = np.ma.array(data['avg']['all'], mask=[invvar == 0])

    # Finally, compute variance weighted average & fill masked entries with 0
    wavg = np.ma.average(data['avg']['all'], weights=invvar, axis=3).filled(0)

    # Export
    if not exname and fdir:
        exname = os.path.split(fdir)[-1]
    if not exname or exname == '.':
        exname = 'wavg'
    else:
        exname = f'wavg_{exname}'
    export_nifti(wavg.astype(float), img, exname)


def compute_metric(data, atlases, mask, metric='avg'):
    """
    Compute a metric (e.g. average) in the parcels of an atlas.

    The metric is computed in the last axis of `data`.
    """
    print(f'Compute metrics: {metric}')
    comp = dict.fromkeys(atlases.keys())
    for atlas in atlases.keys():
        print(f'Working on {atlas}')
        unique = np.unique(atlases[atlas])
        unique = unique[unique > 0]
        print(f'Labels: {unique}, len: {len(unique)}, surr: {data.shape[-1]}')
        # Initialise dataframe and dictionary for series
        parcels = np.empty([len(unique), data.shape[-1]])

        # Compute averages
        for m, label in enumerate(unique):
            print(f'Metric: {metric}, Label: {label} ({m})')
            if metric == 'avg':
                parcels[m, :] = data[atlases[atlas] == label].mean(axis=0)
            elif metric == 'iqr':
                dist = data[atlases[atlas] == label]
                parcels[m, :] = (np.percentile(dist, 75, axis=0) -
                                 np.percentile(dist, 25, axis=0))

        rank = compute_rank(parcels)
        if metric == 'iqr':
            print('Invert iqr rank')
            rank = 100 - rank

        comp[atlas] = atlases[atlas].copy()

        print(f'Recompose atlas {atlas}')
        for m, label in enumerate(unique):
            comp[atlas][atlases[atlas] == label] = rank[m]

    return comp
