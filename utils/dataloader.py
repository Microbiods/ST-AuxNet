from functools import partial
import os
import os.path as osp
from typing import Any, Callable, Optional, NamedTuple, Type
import yaml
import numpy as np
import pandas as pd
from PIL.Image import open as read_image
import torch as t
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import PIL
import torchvision
import glob
import openslide
import pickle
import logging
import pathlib
import statistics
import collections
from . import ensembl





# TODO: make this a visiondataset
class Spatial(torch.utils.data.Dataset):
    def __init__(self,
                 patient=None,
                 transform=None,
                 window=224,
                 load_image=True,
                 root='data/hist2tscript-filter/',
                 gene_filter="none"):

        self.dataset = sorted(glob.glob("{}/*/*/*.npz".format(root)))
        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

            # TODO: if patient == []
            # Could just throw an error

        self.transform = transform
        self.window = window
        self.load_image = load_image
        self.root = root


        with open(root + "/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        # here is all for gene names  
        with open(root + "/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        self.gene_names = list(map(lambda x: ensembl.symbol[x], self.ensg_names))
        self.mean_expression = np.load('data/hist2tscript-filter/mean_expression.npy')
        # self.median_expression = np.load('data/hist2tscript-filter/median_expression.npy')


        self.slide = collections.defaultdict(dict)
        # TODO: this can be parallelized
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}/{}/{}/{}_{}.tif".format('data/hist2tscript', self.subtype[patient], patient, patient, section))

        '''
        There are some standards:
        1. choose the gene with the mean expression more than 0
        2. filtered out genes that are expressed in less than 10% of the all array spots across all the samples
        3. selected spots with at least ten total read counts, delete the spot near the edge

        '''


        if gene_filter is None or gene_filter == "none":
            self.gene_filter = None
            
        elif gene_filter == "high":
            self.gene_filter = np.array([m > 1. for m in self.mean_expression])

        elif isinstance(gene_filter, int):
            keep = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:gene_filter]))[1])
            self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        else:
            raise ValueError()



        if self.gene_filter is not None:


            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, ~self.gene_filter) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, ~self.gene_filter) if f]

            self.ensg_names = [n for (n, f) in zip(self.ensg_names, self.gene_filter) if f]
            self.gene_names = [n for (n, f) in zip(self.gene_names, self.gene_filter) if f]


            self.mean_expression_aux = self.mean_expression[~self.gene_filter]
            self.mean_expression = self.mean_expression[self.gene_filter]

            # self.median_expression = self.median_expression[self.gene_filter]



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])

        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]


        if self.load_image:

            cached_image = "{}/{}/{}/{}_{}_{}_{}_{}.tif".format(self.root, self.subtype[patient], patient, patient, section, self.window, coord[0], coord[1])
            
            if pathlib.Path(cached_image).exists():
                X = PIL.Image.open(cached_image)

            else:
                slide = self.slide[patient][section]
                X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
                X = X.convert("RGB")

            # if self.cache:
                X.save(cached_image)




        if self.transform is not None:
            X = self.transform(X)
       

        Z = np.sum(count)
        n = count.shape[0]


        if self.gene_filter is not None:
            
            aux = count[~self.gene_filter]
            count = count[self.gene_filter]
            


        y = torch.as_tensor(count, dtype=torch.float)
        aux = torch.as_tensor(aux, dtype=torch.float)


        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        y = torch.log(1 + y)
        aux = torch.log(1 + aux)

        # y = torch.log((1 + y) / (n + Z))
        # aux = torch.log((1 + aux) / (n + Z))

        return X, y, aux, coord, index, patient, section, pixel

