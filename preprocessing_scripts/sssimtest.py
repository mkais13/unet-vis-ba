import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize 
import os
import pandas as pd
from skimage import img_as_float
from keras.preprocessing.image import load_img 
import plotly.express as px


optimizers = [
    "Adagrad",
    "SGD",
    "Adam"
]

topologyfactors = [
    0.5, 
    1.0, 
    1.5
]


batchsizes = [
    3, 
    10
]

lossfunctions = [
    "mean_squared_error",
    "binary_crossentropy",
    "msssim"
]

kernelinitializers = [
    "he_normal",
    "he_uniform"
]


def get_identifiers():
    identifiers=[]
    #get all run_identifiers
    for bs in batchsizes:
        for lf in lossfunctions:
            for opt in optimizers:
                for tf in topologyfactors:
                    for ki in kernelinitializers:
                        identifiers.append('bs{0}-lf{1}-opt{2}-tf{3}-ki{4}'.format(bs,lf,opt,tf,ki))
    return identifiers


filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/run3/results"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/test-labels/test-labels-0-256"

identifiers = get_identifiers()
plotdatalist = []

groundtruth_img = imread(os.path.join(groundtruthpath,"test-labels-1.png"),as_gray=True)
groundtruth_img_resized = resize(groundtruth_img, (256, 256))


testimg = load_img(os.path.join(groundtruthpath, "test-labels-1.png"), target_size=(256,256), interpolation="bicubic")


for index in range(len(identifiers)):
    current_img = imread(os.path.join(filepath, identifiers[index], "1" +"_predict.png"),as_gray=True)
    #current_img = load_img(os.path.join(filepath, identifiers[index], "1" +"_predict.png"), target_size=(256,256), interpolation="bicubic")
    plotdatalist.append([identifiers[index]])
    #current_img = np.resize(current_img,groundtruth_img.shape)
    plotdatalist[index].insert(1, ssim(groundtruth_img_resized, current_img))
    #print(current_img.shape)
print(plotdatalist)

plotdatadf = pd.DataFrame(plotdatalist, columns=["id", "ssim"])

fig = px.bar(plotdatadf, x="id", y="ssim")
fig.show()


