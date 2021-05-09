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




# if bigger than the neighbor, then choose the most and devide by the number of the most,




def run_spatial(args=None):

    
        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ### Split patients into folds ###
        patient = get_spatial_patients()


        # randomly choose one patient in train data as the val data
        train_name = list( set(patient.keys()) - set(args.testpatients) )
        val_name = random.sample(train_name, 1)
        train_name = list( set(train_name) - set(val_name) )

        train_patients = []
        val_patients = []
        test_patients = []


        for (i, p) in enumerate(patient):
            for s in patient[p]:
                if p in args.testpatients :
                    test_patients.append((p, s))
                elif p in val_name:
                    val_patients.append((p, s))
                else:
                    train_patients.append((p, s))
    
        print('Train patients: ',train_name)
        print('Val patient: ', val_name)
        print('Test patient: ', args.testpatients[0])


        # train_patients = []
        # test_patients = []
        # for (i, p) in enumerate(patient):
        #     for s in patient[p]:
        #         if p in args.testpatients or (p, s) in args.testpatients:
        #             test_patients.append((p, s))
        #         else:
        #             train_patients.append((p, s))

        # # choose 10% from the sections as the val dataset
        # val_patients = random.sample(train_patients, int(len(train_patients) * 0.1))
        # train_patients = list( set(train_patients) - set(val_patients) )
       

        ### Dataset setup ###  
        window = args.window
        # here need to be changed for split training data and test data, full datasets
        train_dataset = utils.Spatial(train_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        
        val_dataset = utils.Spatial(val_patients, window=window, gene_filter=args.gene_filter, 
                transform=torchvision.transforms.ToTensor())
        
        ### choose 10% from all the spots (data leakage risk), not spatial class anymore
        # len(train_dataset): 29678
        # train_size = int(0.9 * len(full_dataset))
        # val_size = len(full_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        ###

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
                                    num_workers=args.workers, shuffle=True, pin_memory=True)

        
        # Estimate mean and covariance (calculate X, we only need to read the images)
        t = time.time()
        n_samples = 100   # only choose 100 batches
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

        '''
        for multiscale: raw window size should be larger than 224, 
        by using transforms.CenterCrop we can use bigger window size 
        '''
        
        # TTA(test time augmentation)
        # Transform and data argumentation (TODO: spatial and multiscale ensemble)  

        # for training data
        transform = []
        transform.extend([torchvision.transforms.RandomHorizontalFlip(), # default p = 0.5
                          torchvision.transforms.RandomVerticalFlip(),

                          torchvision.transforms.RandomApply(([
                          random.choice([torchvision.transforms.RandomRotation((-90,-90)),
                          torchvision.transforms.RandomRotation((90,90)),
                          torchvision.transforms.RandomRotation((-180,-180)),
                          torchvision.transforms.RandomRotation((180,180)),
                          torchvision.transforms.RandomRotation((-270,-270)),
                          torchvision.transforms.RandomRotation((270,270)) ])])),
                        #   torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)   ])
                          
        transform = torchvision.transforms.Compose(transform)
        train_dataset.transform = transform


        # for val data
        transform = []
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=mean, std=std)])
        val_dataset.transform = transform



        # for test data
        # if args.average:
        transform = []
        # transform = torchvision.transforms.Compose([utils.transforms.EightSymmetry(),
        #                                             torchvision.transforms.Lambda(lambda symmetries: 
        #               torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
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


        ### Model setup ###
        # get the model parameters
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = utils.MultiT.AuxNet(model._fc.in_features, outputs, outputs_aux)


        # print(effi_model)
        # print(torch.nn.Sequential(*list(model.children())[:-4]))
        # feature = model._fc.in_features
        # model._fc = torch.nn.Linear(in_features=feature,out_features=outputs)
        # model = torchvision.models.__dict__[args.model](pretrained=True)
        
        ### Changes number of outputs for the model, return model###
        # model = utils.MultiT.AuxNet(model, outputs, outputs_aux)
        # utils.nn.set_out_features(model, outputs, outputs_aux)
        # model = torch.nn.DataParallel(model)
        model.to(device)
        # print(model)
        ### Optimizer setup ###
        # chose the parameters that need to be optimized  #
        # parameters = utils.nn.get_finetune_parameters(model, 1, True)
        # optim = torch.optim.__dict__[args.optim](model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # Adam
        optim = torch.optim.__dict__['Adam'](model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',factor=0.5, verbose = True, min_lr=1e-6, patience = 5)
        early_stopping = utils.lrsche.EarlyStopping(patience = 10, verbose=True)


        ### Compute mean expression as baseline ###
        
        # t = time.time()
        # mean_expression = torch.zeros(train_dataset[0][1].shape)
        # mean_expression_aux = torch.zeros(train_dataset[0][2].shape)
       
        # for (i, (_, gene, aux, *_)) in enumerate(train_loader):
        #     print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)

        #     mean_expression += torch.sum(gene, 0)/gene.shape[0]
        #     mean_expression_aux += torch.sum(aux, 0)/aux.shape[0]
            
        # mean_expression /= len(train_loader)
        # mean_expression_aux /= len(train_loader)

        # mean_expression = mean_expression.to(device)
        # mean_expression_aux = mean_expression_aux.to(device)

        ### reset the initial parsers for the fc layers

        # if start_epoch == 0:
        #     m = model
        #     last1 = m.classifier1
        #     last2 = m.classifier2

        #     last1.weight.data.zero_()
        #     last1.bias.data = mean_expression.clone()
        #     last2.weight.data.zero_()
        #     last2.bias.data = mean_expression_aux.clone()
        # print("Computing mean expression took {}".format(time.time() - t))



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



parser.add_argument('--seed', '-s', type=int, default=0, help='Set seed for reproduction')


parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="all the rest patients will be used as the training data")


parser.add_argument("--batch", type=int, default=8, help="training batch size")    
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")



parser.add_argument("--window", type=int, default=224, help="window size")
# try 128 150 224 (small, normal, and a little bigger)

# TODO
parser.add_argument("--gene_filter", choices=["none", "high", 250], default=250,
                       help="specific predicted main genes (defalt use all the rest for the aux tasks)")

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