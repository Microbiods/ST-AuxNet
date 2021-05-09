





First, download and compress the dataset from here https://data.mendeley.com/datasets/29ntw7sh4r/2  to the data folder (delete the raw metadata.csv, using ours)

Then run the reset_raw_file.ipynb in data folder from the beginning to the end to prepare the dataset

Then run 'python3 generate.py' to generate necessary files 

Then run 'bash create_tifs.sh' to transform the jpg to tif

Then run 'python3 model_run.py --testpatients BC23450 --pred_root output/ --epochs 1 --model densenet121' to training the model

Then run  'python3 visualize.py output/1.npz --gene FASN' to generate the gene figure for specific patient images (BC23450) and specific gene (FASN)
