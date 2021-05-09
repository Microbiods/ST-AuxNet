#!/bin/bash
for i in data/*/*/*/*_*.jpg;
do
    echo ${i}
    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done


# add a step before transform to tif
