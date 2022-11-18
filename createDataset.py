import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

bart = [f for f in listdir("Treino") if isfile(join("Treino", f)) and "bart" in f]
homer = [f for f in listdir("Treino") if isfile(join("Treino", f)) and "homer" in f]

bartImages = [Image.open(f"Treino/{x}").convert('LA') for x in bart]
homerImages = [Image.open(f"Treino/{x}") for x in homer]

def calculateMeanShape(bartImages, homerImages):
    meanShape = np.array([0,0])
    bartSize = len(bartImages)
    homerSize =len(homerImages)
    for i in range(max(bartSize, homerSize)):
        if i < bartSize:
            meanShape += bartImages[i].size
        if i < homerSize:
            meanShape += homerImages[i].size
    return meanShape / (bartSize + homerSize)

def applyImageAugmentation(bartImages, homerImages):
    meanShape = calculateMeanShape(bartImages, homerImages)
    bartSize = len(bartImages)
    homerSize =len(homerImages)
    for i in range(max(bartSize, homerSize)):
        if i < bartSize:
            bartImages[i] = bartImages[i].resize((64, 64), resample=Image.Resampling.BILINEAR).resize((int(meanShape[0]), int(meanShape[1])), Image.Resampling.NEAREST)
        if i < homerSize:
            #homerImages[i] = homerImages[i].resize((64, 64), resample=Image.Resampling.BILINEAR).resize((int(meanShape[0]), int(meanShape[1])), Image.Resampling.NEAREST)
            homerImages[i] = homerImages[i].resize((int(meanShape[0]), int(meanShape[1])), Image.Resampling.NEAREST)

def aggregateImages(img1, img2):
    totalWidth = img1.size[0] + img2.size[0]
    maxHeight = max(img1.size[1], img2.size[1])
    new_im = Image.new('RGB', (totalWidth, maxHeight))
    new_im.paste(img1, (0,0))
    new_im.paste(img2, (img2.size[0],0))
    return new_im

def generateDataSet(bartImages, homerImages):
    result = []
    for b in bartImages:
        for h in homerImages:
            result.append(aggregateImages(b, h))
    return result

counter = 0
for i in generateDataSet(bartImages, homerImages):
    i.save(f"train/image{counter}.png")
    counter += 1