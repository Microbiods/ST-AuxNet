import os
import numpy as np
import pickle
import logging
import pathlib
import datetime
import time
import glob
import collections
import tqdm
import argparse
from PIL import Image
import staintools
import skimage.io
Image.MAX_IMAGE_PIXELS = 1000000000        





def spatial(args):


    
    window = 224  # only to check if patch is off of boundary
 

    logger = logging.getLogger(__name__)

    pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)


    # return data, subtype(LA, LB, TNBC)
    raw, subtype = load_raw(args.root)

    with open(args.dest + "/subtype.pkl", "wb") as f:   # 1. save subtype
        pickle.dump(subtype, f)

    t = time.time()
    t0 = time.time()

    section_header = None
    gene_names = set()

    for patient in raw:
        for section in raw[patient]:
            section_header = raw[patient][section]["count"].columns.values[0]
            # TODO: here we can change to intersection
            gene_names = gene_names.union(set(raw[patient][section]["count"].columns.values[1:]))

    gene_names = list(gene_names)
    gene_names.sort()  # sort by name
    with open(args.dest + "/gene.pkl", "wb") as f:     # 2. save all gene names (sorted)
        pickle.dump(gene_names, f)
    temp = gene_names
    gene_names = [section_header] + gene_names
    print("Finding list of genes: " + str(time.time() - t0))

    for (i, patient) in enumerate(raw):
        print("Processing " + str(i + 1) + " / " + str(len(raw)) + ": " + patient)

        for section in raw[patient]:

            pathlib.Path("{}/{}/{}".format(args.dest, subtype[patient], patient)).mkdir(parents=True, exist_ok=True)

            # This is just a blank file to indicate that the section has been completely processed.
            # Preprocessing occassionally crashes, and this lets the preparation restart from where it let off
            complete_filename = "{}/{}/{}/.{}".format(args.dest, subtype[patient], patient, section)
            if pathlib.Path(complete_filename).exists():
                print("Patient {} section {} has already been processed.".format(patient, section))
            else:
                print("Processing " + patient + " " + section + "...")

                # In the original data, genes with no expression in a section are dropped from the table.
                # This adds the columns back in so that comparisons across the sections can be done.
                t0 = time.time()
                missing = list(set(gene_names) - set(raw[patient][section]["count"].keys()))
                c = raw[patient][section]["count"].values[:, 1:].astype(float)
                pad = np.zeros((c.shape[0], len(missing)))
                c = np.concatenate((c, pad), axis=1)
                names = np.concatenate((raw[patient][section]["count"].keys().values[1:], np.array(missing)))
                c = c[:, np.argsort(names)]
                print("Adding zeros and ordering columns: " + str(time.time() - t0))

                t0 = time.time()
                count = {}
                for (j, row) in raw[patient][section]["count"].iterrows():
                    count[row.values[0]] = c[j, :]
                print("Extracting counts: " + str(time.time() - t0))

                t0 = time.time()

                # tumor = {}
                # not_int = False
                # for (_, row) in raw[patient][section]["tumor"].iterrows():
                #     if isinstance(row[1], float) or isinstance(row[2], float):
                #         not_int = True
                #     tumor[(int(round(row[1])), int(round(row[2])))] = (row[4] == "tumor")
                # if not_int:
                #     logger.warning("Patient " + patient + " " + section + " has non-integer patch coordinates.")
                # logger.info("Extracting tumors: " + str(time.time() - t0))

                t0 = time.time()
                image = skimage.io.imread(raw[patient][section]["image"])

                print("Loading image: " + str(time.time() - t0))

                # data = []
                for (_, row) in raw[patient][section]["spot"].iterrows():

                    # x = int(round(row["pixel_x"]))
                    # y = int(round(row["pixel_y"]))
                    
         
                    x = int(round(float(row[0].split(',')[1])))   # coord
                    y = int(round(float(row[0].split(',')[2])))

                    spot_x = int(str(row.values[0].split("x")[0])) # spot id
                    spot_y = int(str(row.values[0].split("x")[1].split(',')[0]))    



                    X = image[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]


                    if X.shape == (window, window, 3): 
                    # negative dimensions are not allowed therefore here we make sure all the selected images are not exceeded 
                    

                        if (str(spot_x) + "x" + str(spot_y)) in list(count.keys()) :

                        # We predict the gene expression by images, therefore we must need images
                        # images -> imge spot coord name -> make sure the name in the raw count file = make sure we have the GE data 
                        # for all the images 

                         
                            if np.sum(count[str(spot_x) + "x" + str(spot_y)]) >= 10:
                          
                                filename = "{}/{}/{}/{}_{}_{}.npz".format(args.dest, subtype[patient], patient, section,
                                                                        spot_x, spot_y)
                                np.savez_compressed(filename, count=count[str(spot_x) + "x" + str(spot_y)],
                                                    # tumor=tumor[(int(row["x"]), int(row["y"]))],
                                                    pixel=np.array([x, y]),
                                                    patient=np.array([patient]),
                                                    section=np.array([section]),
                                                    index=np.array([spot_x, spot_y]))
                            else:
                                logger.warning("Total counts of Patch " + str(spot_x) + "x" + str(spot_y) + " in " + patient + " " + section + " less than 10")




                        else:
                            logger.warning("Patch " + str(spot_x) + "x" + str(spot_y) + " not found in " + patient + " " + section)
                    else:
                        logger.warning("Detected spot too close to edge.")



                print("Saving patches: " + str(time.time() - t0))

                with open(complete_filename, "w"):
                    pass
    print("Preprocessing took " + str(time.time() - t) + " seconds")

    if (not os.path.isfile("data/hist2tscript-patch/mean_expression.npy") or
        not os.path.isfile("data/hist2tscript-patch/median_expression.npy")):


        logging.info("Computing statistics of dataset")   # only compute the saved genes
        gene = []
        for filename in tqdm.tqdm(glob.glob("{}/*/*/*_*_*.npz".format(args.dest))):
            npz = np.load(filename)
            count = npz["count"]
            gene.append(np.expand_dims(count, 1))


        gene = np.concatenate(gene, 1)  #（26933, 30625）

        # delete all gene with 0 mean expression across spots
        # keep the gene showed more than 10% of the spots across samples
        # fullfilled two conditions at the same time

        filter = ((np.sum(gene, 1)!=0)*1 + ((np.sum(np.array(gene!=0),1))>=(0.1*gene.shape[1]))*1)==2

        print( "There are {} genes and {} spots in total for all the patients and sections.".format(gene.shape[0], gene.shape[1]))

        np.save( "data/hist2tscript-patch/mean_expression.npy", np.mean(gene, 1))
        np.save("data/hist2tscript-patch/median_expression.npy", np.median(gene, 1))


        pathlib.Path(args.filter).mkdir(parents=True, exist_ok=True)

        with open(args.filter + "filter.pkl", "wb") as f:   
            pickle.dump(filter, f)


    print('Filter and update all the files: ')

    with open(args.filter + "filter.pkl","rb") as f:   
        filter = pickle.load(f)


    # update all the files  ( 2 npys, 2 pkls, all npzs )
    #  print(gene[filter].shape)  (5943, 30625)

    

    with open(args.filter + "/subtype.pkl", "wb") as f:   # 1. subtype
        pickle.dump(subtype, f)

    with open(args.filter + "/gene.pkl", "wb") as f:     #  2. gene names

        gene_names = list(np.array(temp)[filter])
        pickle.dump(gene_names, f)


    # 3. all npzs                               # hist2tscript-patch/Luminal_B/BC23895
    for filename in tqdm.tqdm(glob.glob("{}/*/*/*_*_*.npz".format(args.dest))):

        

        new_path = "{}{}/{}".format(args.filter, filename.split('/')[2], filename.split('/')[3])
             
             
        
        pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)

        npz = np.load(filename)

        # data/hist2tscript-patch/HER2_non_luminal/BC23810
        new_filename = filename.replace(args.dest, args.filter)

        # print(new_filename)

        np.savez_compressed(new_filename, count = npz["count"][filter],                                
                                      pixel= npz["pixel"],
                                      patient=npz["patient"],
                                      section=npz["section"],
                                      index= npz["index"])

    # 4 pys
    if (not os.path.isfile("data/hist2tscript-filter/mean_expression.npy") or
        not os.path.isfile("data/hist2tscript-filter/median_expression.npy")):

        logging.info("Computing statistics of dataset")   # only compute the saved genes
        gene = []
        for filename in tqdm.tqdm(glob.glob("{}/*/*/*_*_*.npz".format(args.filter))):
            npz = np.load(filename)
            count = npz["count"]
            gene.append(np.expand_dims(count, 1))

        gene = np.concatenate(gene, 1)  
        print( "There are {} genes and {} spots left after filtering.".format(gene.shape[0], gene.shape[1]))

        np.save( "data/hist2tscript-filter/mean_expression.npy", np.mean(gene, 1))
        np.save("data/hist2tscript-filter/median_expression.npy", np.median(gene, 1))








def newer_than(file1, file2):
    """
    Returns True if file1 is newer than file2.
    A typical use case is if file2 is generated using file1.
    For example:

    if newer_than(file1, file2):
        # update file2 based on file1
    """
    return os.path.isfile(file1) and (not os.path.isfile(file2) or os.path.getctime(file1) > os.path.getctime(file2))





def load_section(root: str, patient: str, section: str, subtype: str):
    """
    Loads data for one section of a patient.
    """
    import pandas
    import gzip

    file_root = root + "/" + subtype + "/" + patient + "/" + patient + "_" + section

    # image = skimage.io.imread(file_root + ".jpg")
    image = file_root + ".jpg"

    if newer_than(file_root + ".tsv.gz", file_root + ".count.pkl"):
        with gzip.open(file_root + ".tsv.gz", "rb") as f:
            count = pandas.read_csv(f, sep="\t")
        with open(file_root + ".count.pkl", "wb") as f:
            pickle.dump(count, f)
    else:
        with open(file_root + ".count.pkl", "rb") as f:
            count = pickle.load(f)

    if newer_than(file_root + ".spots.gz", file_root + ".spots.pkl"):
        spot = pandas.read_csv(file_root + ".spots.gz", sep="\t")
        with open(file_root + ".spots.pkl", "wb") as f:
            pickle.dump(spot, f)
    else:
        with open(file_root + ".spots.pkl", "rb") as f:
            spot = pickle.load(f)


    return {"image": image, "count": count, "spot": spot}


def load_raw(root: str):
    """
    Loads data for all patients.
    """

    logger = logging.getLogger(__name__)

    # Wildcard search for patients/sections

    images = glob.glob(root + "/*/*/*_*.jpg")

    #   file_root = root + "/" + subtype + "/" + patient + "/" + patient + "_" + section

    # data/hist2tscript/HER2nonluminal/BC23567/    HE_BC23567_E2.jpg

    # Dict mapping patient ID (str) to a list of all sections available for the patient (List[str])  sections: C/D
    patient = collections.defaultdict(list)
    for (p, s) in map(lambda x: x.split("/")[-1][:-4].split("_"), images):
        patient[p].append(s)

    # Dict mapping patient ID (str) to subtype (str)
    subtype = {}
    for (st, p) in map(lambda x: (x.split("/")[-3], x.split("/")[-1][:-4].split("_")[0]), images):
        if p in subtype:
            if subtype[p] != st:
                raise ValueError("Patient {} is the same marked as type {} and {}.".format(p, subtype[p], st))
        else:
            subtype[p] = st

    print("Loading raw data...")
    t = time.time()
    data = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            data[p] = {}
            for s in patient[p]:
                data[p][s] = load_section(root, p, s, subtype[p])
                pbar.update()
    print("Loading raw data took " + str(time.time() - t) + " seconds.")

    return data, subtype


parser = argparse.ArgumentParser(description='Process the paths.')

parser.add_argument('--root',  type=str, default='data/hist2tscript/',
                    help='an integer for the accumulator')     
parser.add_argument('--dest',  type=str, default='data/hist2tscript-patch/',
                    help='an integer for the accumulator')

parser.add_argument('--filter',  type=str, default='data/hist2tscript-filter/',
                    help='an integer for the accumulator')


args = parser.parse_args()

spatial(args)

