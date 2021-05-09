import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import glob
import socket
import argparse
import collections
import utils
from efficientnet_pytorch import EfficientNet


def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        # device = ("cuda" if args.gpu else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Split patients into folds ###
        patient = get_spatial_patients()
        train_patients = []
        test_patients = []
        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients or (p, s) in args.testpatients:
                    test_patients.append((p, s))
                else:
                    train_patients.append((p, s))

        ### Dataset setup ###  

        window = args.window

        # here need to be changed for split training data and test data

        train_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        
        # print(len(train_dataset)) #29678

        # train_size = int(0.9 * len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
        #                             num_workers=args.workers, shuffle=True, pin_memory=True)





        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        print("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t) + 's')



        # Transform and data argumentation (TODO: spatial and multiscale ensemble)

        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)
        # for training data
        train_dataset.transform = transform



        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(train_loader):

        #     print(X.shape)


        # # for val data
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])
        # val_dataset.transform = transform

        # for test data
        # if args.average:

        transform = []
        transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
                                                    torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
    # # else:
    #     transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])

        test_dataset = utils.Spatial(test_patients, transform = transform, window=args.window, gene_filter=args.gene_filter)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(test_loader):

        #     print(X.shape)


        # Find number of required outputs

        outputs = train_dataset[0][1].shape[0]
           


        start_epoch = 0
        # ### Model setup ###
        model = torchvision.models.__dict__[args.model](pretrained=True)
        ###Changes number of outputs for the model, return model###
        utils.nn.set_out_features(model, outputs)
        # model = torch.nn.DataParallel(model)
        model.to(device)
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        parameters = utils.nn.get_finetune_parameters(model, 1, True)
        optim = torch.optim.__dict__[args.optim](parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)

        # model.to(device)
        # # parameters = utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Compute mean expression as baseline
        
        t = time.time()
        mean_expression = torch.zeros(train_dataset[0][1].shape)
       
        for (i, (_, gene, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

            mean_expression += torch.sum(gene, 0)/gene.shape[0]
        mean_expression /= len(train_loader)
        mean_expression = mean_expression.to(device)


        if args.model != "rf":
            m = model
            # if args.gpu:
            #     m = m.module

            if (isinstance(m, torchvision.models.AlexNet) or
                isinstance(m, torchvision.models.VGG)):
                last = m.classifier[-1]
            elif isinstance(m, torchvision.models.DenseNet):
                last = m.classifier
            elif (isinstance(m, torchvision.models.ResNet) or
                    isinstance(m, torchvision.models.Inception3)):
                last = m.fc
            else:
                raise NotImplementedError()

            if start_epoch == 0:
                last.weight.data.zero_()
             
                last.bias.data = mean_expression.clone()
            

        

        print("Computing mean expression took {}".format(time.time() - t))











        ### Training Loop ###   save for every epoch but actually we just need to save the last one
        # save the best model and npz file

        for epoch in range(start_epoch, args.epochs):
            print("Epoch #" + str(epoch + 1))

            # for each epoch, loop train, val and test

            # for (dataset, loader) in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:

            for (dataset, loader) in [("train", train_loader), ("test", test_loader)]:

                t = time.time()

                if dataset == "train":
                    torch.set_grad_enabled(True)
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    model.eval()

                total = 0
                total_mean = 0
                n=0
                genes = []
                predictions = []
                counts = []
                coord = []
                patient = []
                section = []
                pixel = []

                print(dataset + ":")
                for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(loader):
                    

                    counts.append(gene.detach().numpy())
                    coord.append(c.detach().numpy())
                    patient += pat
                    section += s
                    pixel.append(pix.detach().numpy())

                    X = X.to(device)
                    gene = gene.to(device)





                    if dataset == "test":
                        batch, n_sym, c, h, w = X.shape
                        X = X.view(-1, c, h, w)
                    

                    pred = model(X)  # [32, 5943])

                    if dataset == "test":
                        pred = pred.view(batch, n_sym, -1).mean(1)

                    # print("********")
                    # print(pred[0,0:5])

                    predictions.append(pred.cpu().detach().numpy())



                    loss = torch.sum((pred - gene) ** 2) / outputs # in a epoch, one batch average gene loss


                    # print(loss)

                    # print(loss/X.shape[0])

                    total += loss.cpu().detach().numpy()
                    # n += gene.shape[0]  #[32, 250]

                    message = ""
                    message += "Batch: {:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Batch-based Loss={:.9f}".format(loss / gene.shape[0])  # loop each batch, print averaged batch-gene loss
                    print(message)

                    
                    if dataset == "train" :
                        optim.zero_grad()
                        loss.backward()
                        optim.step()  # 一个batch的数据，计算一次梯度，更新一次网络


                print("    Epoch-based Loss:       " + str(total / len(loader.dataset)))
     
                # one epoch finished, and for the last is test loop, we save
                predictions = np.concatenate(predictions)
                counts = np.concatenate(counts)
                coord = np.concatenate(coord)
                pixel = np.concatenate(pixel)
                # me = mean_expression.cpu().numpy(),  # this is training mean_expression
    #                   

                pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.pred_root + str(epoch + 1),
                                    task="gene",
                                    counts=counts,
                                    predictions=predictions,
                                    coord=coord,
                                    patient=patient,
                                    section=section,
                                    pixel=pixel,
                                    # mean_expression=me,
                                    ensg_names=test_dataset.ensg_names,
                                    gene_names=test_dataset.gene_names,
                )

                # Saving after test so that if information from test is needed, they will not get skipped
                # if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                #     pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                  

                #     torch.save({
                #         'model': model.state_dict(),
                #         'optim' : optim.state_dict(),
                #     }, args.checkpoint + str(epoch + 1) + ".pt")

                #     if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                #         os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")


def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/hist2tscript/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

parser = argparse.ArgumentParser(description='Process the paths.')



parser.add_argument('--seed', '-s', type=int, default=0, help='RNG seed')
# parser.add_argument("--gpu", action="store_true", help="use GPU")


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")
parser.add_argument("--window", type=int, default=224, help="window size")
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="special gene filters")
parser.add_argument("--batch", type=int, default=32, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")



parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
# parser.add_argument("--pretrained", action="store_true",
#                     help="use ImageNet pretrained weights")
# parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
#                              help="fine tune last n layers")
# parser.add_argument("--randomize", action="store_true",
#                                    help="randomize weights in layers to be fined tuned")

parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

parser.add_argument("--average", action="store_true", help="average between rotations and reflections")
parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")

args = parser.parse_args()

run_spatial(args)





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
        with open(root + "/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        self.gene_names = list(map(lambda x: ensembl.symbol[x], self.ensg_names))
        self.mean_expression = np.load('data/hist2tscript-filter/mean_expression.npy')
        self.median_expression = np.load('data/hist2tscript-filter/median_expression.npy')


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
            self.ensg_names = [n for (n, f) in zip(self.ensg_names, self.gene_filter) if f]
            self.gene_names = [n for (n, f) in zip(self.gene_names, self.gene_filter) if f]
            self.mean_expression = self.mean_expression[self.gene_filter]
            self.median_expression = self.median_expression[self.gene_filter]



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])

        count   = npz["count"]
        # tumor   = npz["tumor"]
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
            count = count[self.gene_filter]


            # aux = count[~self.gene_filter]


        y = torch.as_tensor(count, dtype=torch.float)
        # aux = torch.as_tensor(aux, dtype=torch.float)

        # tumor = torch.as_tensor([1 if tumor else 0])


        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        y = torch.log((1 + y) / (n + Z))

        # aux = torch.log((1 + aux) / (n + Z))
        
        # y = torch.log(1 + y)
    
        # return X, tumor, y, coord, index, patient, section, pixel, f

        return X, y, coord, index, patient, section, pixel






import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import glob
import socket
import argparse
import collections
import utils
from efficientnet_pytorch import EfficientNet


def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        # device = ("cuda" if args.gpu else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Split patients into folds ###
        patient = get_spatial_patients()
        train_patients = []
        test_patients = []
        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients or (p, s) in args.testpatients:
                    test_patients.append((p, s))
                else:
                    train_patients.append((p, s))

        ### Dataset setup ###  

        window = args.window

        # here need to be changed for split training data and test data

        train_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        
        # print(len(train_dataset)) #29678

        # train_size = int(0.9 * len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
        #                             num_workers=args.workers, shuffle=True, pin_memory=True)





        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        print("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t) + 's')



        # Transform and data argumentation (TODO: spatial and multiscale ensemble)

        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)
        # for training data
        train_dataset.transform = transform



        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(train_loader):

        #     print(X.shape)


        # # for val data
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])
        # val_dataset.transform = transform

        # for test data
        # if args.average:

        transform = []
        transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
                                                    torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
    # # else:
    #     transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])

        test_dataset = utils.Spatial(test_patients, transform = transform, window=args.window, gene_filter=args.gene_filter)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(test_loader):

        #     print(X.shape)


        # Find number of required outputs

        outputs = train_dataset[0][1].shape[0]
        outputs_aux = train_dataset[0][2].shape[0]
        
        combine_dim = (outputs + outputs_aux)
        


        start_epoch = 0
        # ### Model setup ###
        model = torchvision.models.__dict__[args.model](pretrained=True)

        # print(model)
        ###Changes number of outputs for the model, return model###

        model = utils.AuxNet.MtNet(model, outputs, outputs_aux)
        # utils.nn.set_out_features(model, outputs, outputs_aux)

        # model = torch.nn.DataParallel(model)
        model.to(device)

        # print(model)


        
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        # parameters = utils.nn.get_finetune_parameters(model, 1, True)
        optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # optim_aux = torch.optim.__dict__[args.optim](model.fc2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)

        # model.to(device)
        # # parameters = utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Compute mean expression as baseline
        
        t = time.time()
        mean_expression = torch.zeros(train_dataset[0][1].shape)
        mean_expression_aux = torch.zeros(train_dataset[0][2].shape)
       
        for (i, (_, gene, aux, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

            mean_expression += torch.sum(gene, 0)/gene.shape[0]
            mean_expression_aux += torch.sum(aux, 0)/aux.shape[0]
            
        mean_expression /= len(train_loader)
        mean_expression_aux /= len(train_loader)
        mean_expression = mean_expression.to(device)
        mean_expression_aux = mean_expression_aux.to(device)





        # if args.model != "rf":
        #     m = model
        #     # if args.gpu:
        #     #     m = m.module

        #     if (isinstance(m, torchvision.models.AlexNet) or
        #         isinstance(m, torchvision.models.VGG)):
        #         last = m.classifier[-1]
        #     elif isinstance(m, torchvision.models.DenseNet):
        #         last1 = m.fc1
        #         last2 = m.fc2
        #     elif (isinstance(m, torchvision.models.ResNet) or
        #             isinstance(m, torchvision.models.Inception3)):
        #         last = m.fc
        #     else:
        #         raise NotImplementedError()
        
        if start_epoch == 0:
            m = model
            last1 = m.classifier1
            last2 = m.classifier2

            last1.weight.data.zero_()
            
            last1.bias.data = mean_expression.clone()

            last2.weight.data.zero_()
            
            last2.bias.data = mean_expression_aux.clone()
        

        

        print("Computing mean expression took {}".format(time.time() - t))











        ### Training Loop ###   save for every epoch but actually we just need to save the last one
        # save the best model and npz file

        for epoch in range(start_epoch, args.epochs):
            print("Epoch #" + str(epoch + 1))

            # for each epoch, loop train, val and test

            # for (dataset, loader) in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:

            for (dataset, loader) in [("train", train_loader), ("test", test_loader)]:

                t = time.time()

                if dataset == "train":
                    torch.set_grad_enabled(True)
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    model.eval()

                total = 0
                total_mean = 0
                n=0
                genes = []
                predictions = []
                predictions_aux=[]
                counts = []
                coord = []
                patient = []
                section = []
                pixel = []

                print(dataset + ":")
                for (i, (X, gene, aux, c, ind, pat, s, pix)) in enumerate(loader):
                    

                    counts.append(gene.detach().numpy())
                    coord.append(c.detach().numpy())
                    patient += pat
                    section += s
                    pixel.append(pix.detach().numpy())

                    X = X.to(device)
                    gene = gene.to(device)

                    aux= aux.to(device)





                    if dataset == "test":
                        batch, n_sym, c, h, w = X.shape
                        X = X.view(-1, c, h, w)
                    

                    # pred = model.fc1(X)  # [32, 5943])

                    # pred_aux = model.fc2(X)

                    pred, pred_aux =model(X)

                    if dataset == "test":
                        pred = pred.view(batch, n_sym, -1).mean(1)

                        pred_aux = pred_aux.view(batch, n_sym, -1).mean(1)

                    # print("********")
                    # print(pred[0,0:5])

                    predictions.append(pred.cpu().detach().numpy())

                    predictions_aux.append(pred.cpu().detach().numpy())



                    loss = torch.sum((pred - gene) ** 2) / outputs # in a epoch, one batch average gene loss

                    loss_aux = torch.sum((pred_aux - aux) ** 2) / outputs 

                    loss_comb = loss + loss_aux

                    # print(loss)

                    # print(loss/X.shape[0])

                    total += loss.cpu().detach().numpy()
                    # n += gene.shape[0]  #[32, 250]

                    message = ""
                    message += "Batch: {:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Batch-based main Loss={:.9f}".format(loss / gene.shape[0])  # loop each batch, print averaged batch-gene loss

                    message += "    Batch-based aux Loss={:.9f}".format(loss_aux / gene.shape[0])

                    message += "    Batch-based cimbine Loss={:.9f}".format(loss_comb / gene.shape[0])
                    print(message)

                    
                    if dataset == "train" :
                        optim.zero_grad()
                       
                        loss_comb.backward()
                        optim.step()  # 一个batch的数据，计算一次梯度，更新一次网络
                    

                print("    Epoch-based Loss:       " + str(total / len(loader.dataset)))
     
                # one epoch finished, and for the last is test loop, we save
                predictions = np.concatenate(predictions)
                predictions_aux = np.concatenate(predictions_aux)
                
                counts = np.concatenate(counts)
                coord = np.concatenate(coord)
                pixel = np.concatenate(pixel)
                # me = mean_expression.cpu().numpy(),  # this is training mean_expression
    #                   

                pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.pred_root + str(epoch + 1),
                                    task="gene",
                                    counts=counts,
                                    predictions=predictions,
                                    predictions_aux=predictions_aux,
                                    coord=coord,
                                    patient=patient,
                                    section=section,
                                    pixel=pixel,
                                    # mean_expression=me,
                                    ensg_names=test_dataset.ensg_names,
                                    gene_names=test_dataset.gene_names,
                )

                # Saving after test so that if information from test is needed, they will not get skipped
                # if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                #     pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                  

                #     torch.save({
                #         'model': model.state_dict(),
                #         'optim' : optim.state_dict(),
                #     }, args.checkpoint + str(epoch + 1) + ".pt")

                #     if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                #         os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")


def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/hist2tscript/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

parser = argparse.ArgumentParser(description='Process the paths.')



parser.add_argument('--seed', '-s', type=int, default=0, help='RNG seed')
# parser.add_argument("--gpu", action="store_true", help="use GPU")


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")
parser.add_argument("--window", type=int, default=224, help="window size")
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="special gene filters")
parser.add_argument("--batch", type=int, default=32, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")



parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
# parser.add_argument("--pretrained", action="store_true",
#                     help="use ImageNet pretrained weights")
# parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
#                              help="fine tune last n layers")
# parser.add_argument("--randomize", action="store_true",
#                                    help="randomize weights in layers to be fined tuned")

parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

parser.add_argument("--average", action="store_true", help="average between rotations and reflections")
parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")

args = parser.parse_args()

run_spatial(args)






import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import glob
import socket
import argparse
import collections
import utils
from efficientnet_pytorch import EfficientNet


def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        # device = ("cuda" if args.gpu else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Split patients into folds ###
        patient = get_spatial_patients()
        train_patients = []
        test_patients = []
        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients or (p, s) in args.testpatients:
                    test_patients.append((p, s))
                else:
                    train_patients.append((p, s))

        ### Dataset setup ###  

        window = args.window

        # here need to be changed for split training data and test data

        train_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        
        # print(len(train_dataset)) #29678

        # train_size = int(0.9 * len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
        #                             num_workers=args.workers, shuffle=True, pin_memory=True)





        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        print("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t) + 's')



        # Transform and data argumentation (TODO: spatial and multiscale ensemble)

        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)
        # for training data
        train_dataset.transform = transform



        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(train_loader):

        #     print(X.shape)


        # # for val data
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])
        # val_dataset.transform = transform

        # for test data
        # if args.average:

        transform = []
        transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
                                                    torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
    # # else:
    #     transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])

        test_dataset = utils.Spatial(test_patients, transform = transform, window=args.window, gene_filter=args.gene_filter)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # print(transform)

        # for (i, (X, gene, c, ind, pat, s, pix)) in enumerate(test_loader):

        #     print(X.shape)


        # Find number of required outputs

        outputs = train_dataset[0][1].shape[0]
        outputs_aux = train_dataset[0][2].shape[0]
        
        combine_dim = (outputs + outputs_aux)
        


        start_epoch = 0
        # ### Model setup ###
        model = torchvision.models.__dict__[args.model](pretrained=True)

        # print(model)
        ###Changes number of outputs for the model, return model###

        model = utils.AuxNet.MtNet(model, outputs, outputs_aux)
        # utils.nn.set_out_features(model, outputs, outputs_aux)

        # model = torch.nn.DataParallel(model)
        model.to(device)

        # print(model)


        
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        # parameters = utils.nn.get_finetune_parameters(model, 1, True)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # Adam
        
        optim = torch.optim.__dict__['Adam'](model.parameters(), lr=1e-6, weight_decay=args.weight_decay)

        # optim_aux = torch.optim.__dict__[args.optim](model.fc2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)

        # model.to(device)
        # # parameters = utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Compute mean expression as baseline
        
        t = time.time()
        mean_expression = torch.zeros(train_dataset[0][1].shape)
        mean_expression_aux = torch.zeros(train_dataset[0][2].shape)
       
        for (i, (_, gene, aux, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

            mean_expression += torch.sum(gene, 0)/gene.shape[0]
            mean_expression_aux += torch.sum(aux, 0)/aux.shape[0]
            
        mean_expression /= len(train_loader)
        mean_expression_aux /= len(train_loader)
        mean_expression = mean_expression.to(device)
        mean_expression_aux = mean_expression_aux.to(device)





        # if args.model != "rf":
        #     m = model
        #     # if args.gpu:
        #     #     m = m.module

        #     if (isinstance(m, torchvision.models.AlexNet) or
        #         isinstance(m, torchvision.models.VGG)):
        #         last = m.classifier[-1]
        #     elif isinstance(m, torchvision.models.DenseNet):
        #         last1 = m.fc1
        #         last2 = m.fc2
        #     elif (isinstance(m, torchvision.models.ResNet) or
        #             isinstance(m, torchvision.models.Inception3)):
        #         last = m.fc
        #     else:
        #         raise NotImplementedError()
        
        if start_epoch == 0:
            m = model
            last1 = m.classifier1
            last2 = m.classifier2

            last1.weight.data.zero_()
            
            last1.bias.data = mean_expression.clone()

            last2.weight.data.zero_()
            
            last2.bias.data = mean_expression_aux.clone()
        

        

        print("Computing mean expression took {}".format(time.time() - t))











        ### Training Loop ###   save for every epoch but actually we just need to save the last one
        # save the best model and npz file

        for epoch in range(start_epoch, args.epochs):
            print("Epoch #" + str(epoch + 1))

            # for each epoch, loop train, val and test

            # for (dataset, loader) in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:

            for (dataset, loader) in [("train", train_loader), ("test", test_loader)]:

                t = time.time()

                if dataset == "train":
                    torch.set_grad_enabled(True)
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    model.eval()

                total = 0
                total_mean = 0
                n=0
                genes = []
                predictions = []
                predictions_aux=[]
                counts = []
                coord = []
                patient = []
                section = []
                pixel = []

                print(dataset + ":")
                for (i, (X, gene, aux, c, ind, pat, s, pix)) in enumerate(loader):
                    

                    counts.append(gene.detach().numpy())
                    coord.append(c.detach().numpy())
                    patient += pat
                    section += s
                    pixel.append(pix.detach().numpy())

                    X = X.to(device)
                    gene = gene.to(device)

                    aux= aux.to(device)





                    if dataset == "test":
                        batch, n_sym, c, h, w = X.shape
                        X = X.view(-1, c, h, w)
                    

                    # pred = model.fc1(X)  # [32, 5943])

                    # pred_aux = model.fc2(X)

                    pred, pred_aux =model(X)

                    if dataset == "test":
                        pred = pred.view(batch, n_sym, -1).mean(1)

                        pred_aux = pred_aux.view(batch, n_sym, -1).mean(1)

                    # print("********")
                    # print(pred[0,0:5])

                    predictions.append(pred.cpu().detach().numpy())

                    predictions_aux.append(pred.cpu().detach().numpy())



                    loss = torch.sum((pred - gene) ** 2) / outputs # in a epoch, one batch average gene loss

                    loss_aux = torch.sum((pred_aux - aux) ** 2) / outputs 

                    loss_comb = 10*loss + 1*loss_aux

                    # print(loss)

                    # print(loss/X.shape[0])

                    total += loss.cpu().detach().numpy()
                    # n += gene.shape[0]  #[32, 250]

                    message = ""
                    message += "Batch: {:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Batch-based main Loss={:.9f}".format(loss / gene.shape[0])  # loop each batch, print averaged batch-gene loss

                    message += "    Batch-based aux Loss={:.9f}".format(loss_aux / gene.shape[0])

                    message += "    Batch-based cimbine Loss={:.9f}".format(loss_comb / gene.shape[0])
                    print(message)

                    
                    if dataset == "train" :
                        optim.zero_grad()
                       
                        loss_comb.backward()
                        optim.step()  # 一个batch的数据，计算一次梯度，更新一次网络
                    

                print("    Epoch-based Loss:       " + str(total / len(loader.dataset)))
     
                # one epoch finished, and for the last is test loop, we save
                predictions = np.concatenate(predictions)
                predictions_aux = np.concatenate(predictions_aux)
                
                counts = np.concatenate(counts)
                coord = np.concatenate(coord)
                pixel = np.concatenate(pixel)
                # me = mean_expression.cpu().numpy(),  # this is training mean_expression
    #                   

                pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.pred_root + str(epoch + 1),
                                    task="gene",
                                    counts=counts,
                                    predictions=predictions,
                                    predictions_aux=predictions_aux,
                                    coord=coord,
                                    patient=patient,
                                    section=section,
                                    pixel=pixel,
                                    # mean_expression=me,
                                    ensg_names=test_dataset.ensg_names,
                                    gene_names=test_dataset.gene_names,
                )

                # Saving after test so that if information from test is needed, they will not get skipped
                # if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                #     pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                  

                #     torch.save({
                #         'model': model.state_dict(),
                #         'optim' : optim.state_dict(),
                #     }, args.checkpoint + str(epoch + 1) + ".pt")

                #     if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                #         os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")


def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/hist2tscript/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

parser = argparse.ArgumentParser(description='Process the paths.')



parser.add_argument('--seed', '-s', type=int, default=0, help='RNG seed')
# parser.add_argument("--gpu", action="store_true", help="use GPU")


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")
parser.add_argument("--window", type=int, default=224, help="window size")
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="special gene filters")
parser.add_argument("--batch", type=int, default=32, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")



parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
# parser.add_argument("--pretrained", action="store_true",
#                     help="use ImageNet pretrained weights")
# parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
#                              help="fine tune last n layers")
# parser.add_argument("--randomize", action="store_true",
#                                    help="randomize weights in layers to be fined tuned")



parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")





parser.add_argument("--epochs", type=int, default=50, help="number of epochs")




parser.add_argument("--average", action="store_true", help="average between rotations and reflections")
parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")

args = parser.parse_args()

run_spatial(args)



# python3 model_run.py --testpatients BC23450 --pred_root output/ --epochs 50 --model densenet121

# python3 visualize.py output/3.npz --gene FASN






import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import glob
import socket
import argparse
import collections
import utils
from efficientnet_pytorch import EfficientNet


def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ### Split patients into folds ###
        patient = get_spatial_patients()
        train_patients = []
        test_patients = []
        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients or (p, s) in args.testpatients:
                    test_patients.append((p, s))
                else:
                    train_patients.append((p, s))

        ### Dataset setup ###  
        window = args.window
        # here need to be changed for split training data and test data, full datasets
        full_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        # print(len(train_dataset)) #29678
       

        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        
        # not spatial class anymore


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)
        
       

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        print("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t) + 's')



        # Transform and data argumentation (TODO: spatial and multiscale ensemble)
        
        # for training data
        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)
        
        # print(train_dataset.transform)
        train_dataset.dataset.transform = transform
        # print(train_dataset.transform)


        # for val data
        transform = []
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=mean, std=std)])
        val_dataset.dataset.transform = transform

        # for test data
        # if args.average:
        transform = []
        transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
                                                    torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
    # # else:
    #     transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize(mean=mean, std=std)])

        test_dataset = utils.Spatial(test_patients, transform = transform, window=args.window, gene_filter=args.gene_filter)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=True)


        # Find number of required outputs
        outputs = train_dataset[0][1].shape[0]
        outputs_aux = train_dataset[0][2].shape[0]
        combine_dim = (outputs + outputs_aux)
        start_epoch = 0


        # ### Model setup ###
        model = torchvision.models.__dict__[args.model](pretrained=True)
        ### Changes number of outputs for the model, return model###
        model = utils.AuxNet.MtNet(model, outputs, outputs_aux)
        # utils.nn.set_out_features(model, outputs, outputs_aux)
        # model = torch.nn.DataParallel(model)
        model.to(device)

        
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        # parameters = utils.nn.get_finetune_parameters(model, 1, True)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # Adam
        optim = torch.optim.__dict__['Adam'](model.parameters(), lr=1e-6, weight_decay=args.weight_decay)



        lr_scheduler = utils.lrsche.LRScheduler(optim)

        early_stopping = utils.lrsche.EarlyStopping()

        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)
        # model.to(device)
        # # parameters = utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        ### Compute mean expression as baseline ###
        
        t = time.time()

        mean_expression = torch.zeros(train_dataset[0][1].shape)
        mean_expression_aux = torch.zeros(train_dataset[0][2].shape)
       
        for (i, (_, gene, aux, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

            mean_expression += torch.sum(gene, 0)/gene.shape[0]
            mean_expression_aux += torch.sum(aux, 0)/aux.shape[0]
            
        mean_expression /= len(train_loader)
        mean_expression_aux /= len(train_loader)
        
        # mean_expression = train_dataset.mean_expression
        # mean_expression_aux = train_dataset.mean_expression_aux

        mean_expression = mean_expression.to(device)
        mean_expression_aux = mean_expression_aux.to(device)

        ### reset the initial parsers for the fc layers

        # if args.model != "rf":
        #     m = model
        #     # if args.gpu:
        #     #     m = m.module
        #     if (isinstance(m, torchvision.models.AlexNet) or
        #         isinstance(m, torchvision.models.VGG)):
        #         last = m.classifier[-1]
        #     elif isinstance(m, torchvision.models.DenseNet):
        #         last1 = m.fc1
        #         last2 = m.fc2
        #     elif (isinstance(m, torchvision.models.ResNet) or
        #             isinstance(m, torchvision.models.Inception3)):
        #         last = m.fc
        #     else:
        #         raise NotImplementedError()
        
        if start_epoch == 0:
            m = model
            last1 = m.classifier1
            last2 = m.classifier2

            last1.weight.data.zero_()
            last1.bias.data = mean_expression.clone()
            last2.weight.data.zero_()
            last2.bias.data = mean_expression_aux.clone()
        

        print("Computing mean expression took {}".format(time.time() - t))



        ### Training Loop ###   save for every epoch

        for epoch in range(start_epoch, args.epochs):
            print("Epoch #" + str(epoch + 1))

            for (dataset, loader) in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                
                t = time.time()

                if dataset == "train":
                    torch.set_grad_enabled(True)
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    model.eval()

                total = 0
                predictions = []
                predictions_aux=[]
                counts = []
                coord = []
                patient = []
                section = []
                pixel = []
                
                val_epoch_loss = []

                print(dataset + ":")

                for (i, (X, gene, aux, c, ind, pat, s, pix)) in enumerate(loader):
                    
                    counts.append(gene.detach().numpy())
                    coord.append(c.detach().numpy())
                    patient += pat
                    section += s
                    pixel.append(pix.detach().numpy())
                    X = X.to(device)
                    gene = gene.to(device)
                    aux= aux.to(device)

                    if dataset == "test":
                        batch, n_sym, c, h, w = X.shape
                        X = X.view(-1, c, h, w)

                    pred, pred_aux =model(X)

                    if dataset == "test":
                        pred = pred.view(batch, n_sym, -1).mean(1)
                        pred_aux = pred_aux.view(batch, n_sym, -1).mean(1)


                    predictions.append(pred.cpu().detach().numpy())
                    predictions_aux.append(pred_aux.cpu().detach().numpy())

                    loss = torch.sum((pred - gene) ** 2) / outputs    #  one batch loss for mean gene expression

                    loss_aux = torch.sum((pred_aux - aux) ** 2) / outputs_aux 

                    loss_comb = loss + 0.5*loss_aux

                    n = gene.shape[0]

                 
                    total += loss.cpu().detach().numpy()


                    # loss for 
                    message = ""
                    message += "Batch (time): {:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Batch-based main Loss={:.9f}".format(loss / n)  # loop each batch, print averaged batch-gene loss
                    message += "    Batch-based aux Loss={:.9f}".format(loss_aux / n)
                    message += "    Batch-based cimbine Loss={:.9f}".format(loss_comb / n)
                    print(message)

                    
                    if dataset == "train" :
                        optim.zero_grad()
                        loss_comb.backward()
                        optim.step()          # for each batch, calculate grads and updates the parsers
                    

                print("    Epoch-based main Loss:       " + str(total / len(loader.dataset)))


                if dataset == "test" :

                        # once epoch finishes, the last dataloader is test data, we save the predictions
                        predictions = np.concatenate(predictions)
                        predictions_aux = np.concatenate(predictions_aux)
                        
                        counts = np.concatenate(counts)
                        coord = np.concatenate(coord)
                        pixel = np.concatenate(pixel)
                            

                        pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.pred_root + str(epoch + 1),
                                            task="gene",
                                            counts=counts,
                                            predictions=predictions,
                                            predictions_aux=predictions_aux,
                                            coord=coord,
                                            patient=patient,
                                            section=section,
                                            pixel=pixel,
                                            ensg_names=test_dataset.ensg_names,
                                            gene_names=test_dataset.gene_names,
                                            ensg_aux=test_dataset.ensg_aux,
                                            gene_aux=test_dataset.gene_aux)

              
                if dataset == "val":
                # loss for each batch
                    val_epoch_loss = total / len(loader.dataset)
                    lr_scheduler(val_epoch_loss)
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        break
                # Saving after test so that if information from test is needed, they will not get skipped
                # if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                #     pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                  

                #     torch.save({
                #         'model': model.state_dict(),
                #         'optim' : optim.state_dict(),
                #     }, args.checkpoint + str(epoch + 1) + ".pt")

                #     if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                #         os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")
           

def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/hist2tscript/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

parser = argparse.ArgumentParser(description='Process the paths.')



parser.add_argument('--seed', '-s', type=int, default=0, help='RNG seed')
# parser.add_argument("--gpu", action="store_true", help="use GPU")


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")
parser.add_argument("--window", type=int, default=224, help="window size")
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="special gene filters")
parser.add_argument("--batch", type=int, default=32, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")


parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
# parser.add_argument("--pretrained", action="store_true",
#                     help="use ImageNet pretrained weights")
# parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
#                              help="fine tune last n layers")
# parser.add_argument("--randomize", action="store_true",
#                                    help="randomize weights in layers to be fined tuned")


parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")

parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

parser.add_argument("--average", action="store_true", help="average between rotations and reflections")
parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")

args = parser.parse_args()

run_spatial(args)


# python3 model_run.py --testpatients BC23450 --pred_root output/ --epochs 50 --model densenet121

# python3 visualize.py output/3.npz --gene FASN






import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import glob
import socket
import argparse
import collections
import utils
from efficientnet_pytorch import EfficientNet


def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ### Split patients into folds ###
        patient = get_spatial_patients()
        train_patients = []
        test_patients = []
        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients or (p, s) in args.testpatients:
                    test_patients.append((p, s))
                else:
                    train_patients.append((p, s))

        ### Dataset setup ###  
        window = args.window
        # here need to be changed for split training data and test data, full datasets
        full_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())

        # print(len(train_dataset)) #29678

        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        

        # data leakage risk


        # not spatial class anymore
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # Estimate mean and covariance
        t = time.time()
        n_samples = 100
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        print("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t) + 's')



        # Transform and data argumentation (TODO: spatial and multiscale ensemble)
        
        # for training data
        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)
        train_dataset.dataset.transform = transform


        # for val data
        transform = []
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=mean, std=std)])
        val_dataset.dataset.transform = transform

        # for test data
        # if args.average:
        transform = []
        # transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
        #                                             torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
    # # else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=mean, std=std)])

        test_dataset = utils.Spatial(test_patients, transform = transform, window=args.window, gene_filter=args.gene_filter)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=True)


        # Find number of required outputs (gene)
        outputs = train_dataset[0][1].shape[0]
        outputs_aux = train_dataset[0][2].shape[0]
        combine_dim = (outputs + outputs_aux)
        start_epoch = 0


        # ### Model setup ###
        model = torchvision.models.__dict__[args.model](pretrained=True)
        ### Changes number of outputs for the model, return model###
        model = utils.AuxNet.MtNet(model, outputs, outputs_aux)
        # utils.nn.set_out_features(model, outputs, outputs_aux)
        # model = torch.nn.DataParallel(model)
        model.to(device)

        
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        # parameters = utils.nn.get_finetune_parameters(model, 1, True)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # Adam
        optim = torch.optim.__dict__['Adam'](model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',factor=0.5, verbose = True, min_lr=1e-6, patience = 5)
        early_stopping = utils.lrsche.EarlyStopping(patience = 10, verbose=True)
        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)
        # model.to(device)
        # # parameters = utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        ### Compute mean expression as baseline ###
        
        t = time.time()
        mean_expression = torch.zeros(train_dataset[0][1].shape)
        mean_expression_aux = torch.zeros(train_dataset[0][2].shape)
       
        for (i, (_, gene, aux, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

            mean_expression += torch.sum(gene, 0)/gene.shape[0]
            mean_expression_aux += torch.sum(aux, 0)/aux.shape[0]
            
        mean_expression /= len(train_loader)
        mean_expression_aux /= len(train_loader)

        mean_expression = mean_expression.to(device)
        mean_expression_aux = mean_expression_aux.to(device)

        ### reset the initial parsers for the fc layers

        if start_epoch == 0:
            m = model
            last1 = m.classifier1
            last2 = m.classifier2

            last1.weight.data.zero_()
            last1.bias.data = mean_expression.clone()
            last2.weight.data.zero_()
            last2.bias.data = mean_expression_aux.clone()
        print("Computing mean expression took {}".format(time.time() - t))



        ### Training Loop ###   save for every epoch

        for epoch in range(start_epoch, args.epochs):
            print("Epoch #" + str(epoch + 1))
            # within the epoch

            for (dataset, loader) in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                # within the dataloader
                t = time.time()

                if dataset == "train":
                    torch.set_grad_enabled(True)
                    model.train()
                else:
                    torch.set_grad_enabled(False)
                    model.eval()

                # for each loader

                total = 0
                predictions = []
                predictions_aux=[]
                counts = []
                coord = []
                patient = []
                section = []
                pixel = []

                print(dataset + ":")

                for (i, (X, gene, aux, c, ind, pat, s, pix)) in enumerate(loader):

                    # within the batch in each loader

                    counts.append(gene.detach().numpy())
                    coord.append(c.detach().numpy())
                    patient += pat
                    section += s
                    pixel.append(pix.detach().numpy())
                    X = X.to(device)
                    gene = gene.to(device)
                    aux= aux.to(device)

                    # if dataset == "test":
                    #     batch, n_sym, c, h, w = X.shape
                    #     X = X.view(-1, c, h, w)

                    pred, pred_aux = model(X)

                    # if dataset == "test":
                    #     pred = pred.view(batch, n_sym, -1).mean(1)
                    #     pred_aux = pred_aux.view(batch, n_sym, -1).mean(1)


                    predictions.append(pred.cpu().detach().numpy())
                    predictions_aux.append(pred_aux.cpu().detach().numpy())

                    loss = torch.sum((pred - gene) ** 2) / outputs    #  one batch loss for mean gene expression

                    loss_aux = torch.sum((pred_aux - aux) ** 2) / outputs_aux 

                    loss_comb = loss + 0.5 * loss_aux

                    n = gene.shape[0]  # size for each batch

                    total += loss.cpu().detach().numpy()

                    batch_loss = loss_comb / n

                    # mean loss for each batch
                    message = ""
                    message += "Batch (time): {:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Batch-based main Loss={:.9f}".format(loss / n)  # loop each batch, print averaged batch-gene loss
                    message += "    Batch-based aux Loss={:.9f}".format(loss_aux / n)
                    message += "    Batch-based combine Loss={:.9f}".format(batch_loss)
                    print(message)
                    
                 

                    # inside batch, inside loader

                    if dataset == "train" :
                        optim.zero_grad()
                        batch_loss.backward()
                        optim.step()          # for each batch, calculate grads and updates the parsers


                # for each loader
                print("    Epoch-based main Loss:       " + str(total / len(loader.dataset)))

                # outside batch, inside loader

                if dataset == 'val':
                    val_epoch_loss = total/ len(loader.dataset)


                if dataset == 'test':
                    # once epoch finishes, the last dataloader is test data, we save the predictions
                    predictions = np.concatenate(predictions)
                    predictions_aux = np.concatenate(predictions_aux)
                    
                    counts = np.concatenate(counts)
                    coord = np.concatenate(coord)
                    pixel = np.concatenate(pixel)
                        

                    pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.pred_root + str(epoch + 1),
                                        task="gene",
                                        counts=counts,
                                        predictions=predictions,
                                        predictions_aux=predictions_aux,
                                        coord=coord,
                                        patient=patient,
                                        section=section,
                                        pixel=pixel,
                                        ensg_names=test_dataset.ensg_names,
                                        gene_names=test_dataset.gene_names,
                                        ensg_aux=test_dataset.ensg_aux,
                                        gene_aux=test_dataset.gene_aux)

                    pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model': model.state_dict(),
                        'optim' : optim.state_dict(),
                    }, args.pred_root + str(epoch + 1) + ".pt")

            # outside loader, inside epoch
            # after each epoch (include train, val, and test), see validation data, to decide learning rate and early stopping

            scheduler.step(val_epoch_loss)
            early_stopping(val_epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

             
           

def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/hist2tscript/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

parser = argparse.ArgumentParser(description='Process the paths.')



parser.add_argument('--seed', '-s', type=int, default=0, help='RNG seed')
# parser.add_argument("--gpu", action="store_true", help="use GPU")


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")
parser.add_argument("--window", type=int, default=224, help="window size")
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="special gene filters")
parser.add_argument("--batch", type=int, default=32, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")


parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
# parser.add_argument("--pretrained", action="store_true",
#                     help="use ImageNet pretrained weights")
# parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
#                              help="fine tune last n layers")
# parser.add_argument("--randomize", action="store_true",
#                                    help="randomize weights in layers to be fined tuned")


parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")

parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

parser.add_argument("--average", action="store_true", help="average between rotations and reflections")
parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")

args = parser.parse_args()

run_spatial(args)


# python3 model_run.py --testpatients BC23450 --pred_root output/ --epochs 50 --model densenet121

# python3 visualize.py output/3.npz --gene FASN