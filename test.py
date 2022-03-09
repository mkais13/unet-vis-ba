from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/UNet/unet/data/membrane/train/label"




img = load_img(os.path.join(groundtruthpath, "2.png"), target_size=(224,224), interpolation="bilinear")
img = np.array(img) 
reshaped_img = img.reshape(1,224,224,3)
print(reshaped_img)
imgplot = plt.imshow(reshaped_img)
plt.show()
