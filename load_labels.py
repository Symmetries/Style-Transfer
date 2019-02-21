from pycocotools.coco import COCO
import torch
import numpy as np

# Load the COCO dataset files
dataDir = '..'
dataType = 'val2014'
fileName = 'validation'
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

numCats = 90
numImages = len(coco.getImgIds())

categories = [[0.0 for _ in range(numCats)] for _ in range(numImages)]

# Create the labels for each image (1 if image is of this category 0 otherwise)
for i, imgId in enumerate(coco.getImgIds()):
    for ann in coco.loadAnns(coco.getAnnIds(imgId)):
        categories[i][ann['category_id']-1] = 1.0

# Save the categories
torch.save(torch.tensor(categories), '{}.pt'.format(fileName))
