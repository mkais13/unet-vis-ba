import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize 
import os
from vggtest import get_identifiers
import pandas as pd
from skimage import img_as_float
from keras.preprocessing.image import load_img 
import plotly.express as px

filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/results"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/UNet/unet/data/membrane/train/label"

identifiers = get_identifiers()
plotdatalist = []

groundtruth_img = imread(os.path.join(groundtruthpath, "1.png"),as_gray=True)
groundtruth_img_resized = resize(groundtruth_img, (256, 256))


testimg = load_img(os.path.join(groundtruthpath, "1.png"), target_size=(256,256), interpolation="bicubic")


for index in range(len(identifiers)):
    #current_img = imread(os.path.join(filepath, identifiers[index], "1" +"_predict.png"),as_gray=True)
    current_img = load_img(os.path.join(filepath, identifiers[index], "1" +"_predict.png"), target_size=(256,256), interpolation="bicubic")
    plotdatalist.append([identifiers[index]])
    #current_img = np.resize(current_img,groundtruth_img.shape)
    plotdatalist[index].insert(1, ssim(testimg, current_img))
    #print(current_img.shape)
print(plotdatalist)

plotdatadf = pd.DataFrame(plotdatalist, columns=["id", "ssim"])

fig = px.bar(plotdatadf, x="id", y="ssim")
fig.show()