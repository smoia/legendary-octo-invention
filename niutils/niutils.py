#!/usr/bin/env python3

import argparse
import os
import sys

import nibabel as nib
import numpy as np


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
