# T. Kowalski project 
# made for Machine Learning, KISD/IFJ PAN, Krakow
# trained network output visualisation

import numpy as np
import tensorflow as tf
import os, glob
from PIL import Image
import matplotlib.pyplot as plt

# data preparation

def input_tensor_prepare(input):
    val=[]
    # image=Image.open(f"/home/geniusz/nn/hires/dataset_hires/original/"+filename)
    image=np.asanyarray(input,dtype=np.float16)
    image=(image-np.min(image))/(np.max(image)-np.min(image))
    # print(np.shape(image))
    val.append(image)
# output=tf.cast(output, dtype=tf.float32)

    return val
    
obraz=1
# os.chdir(r"C:/Users/TK/Desktop/spider_nn/data_generator/datanew/test/images/") # raw data
# # os.chdir(r"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/train/images") # generated data

# # data processing part...

# extension="*.bmp"
# for file in glob.glob(extension):
file=r"/home/geniusz/nn/hires/dataset_hires/original/IMG_20200306_154022.jpg"
input_image_im=Image.open(file)

# preprocessed_tensor=[]
imm=np.expand_dims(np.asarray(input_image_im),axis=0)

# preprocessed_tensor.append(imm)
preprocessed_tensor=input_tensor_prepare(imm)

# loading trained model...

reconstructed_model_1 = tf.keras.models.load_model(r"/home/geniusz/nn/hires/dataset_hires/original/model_epoch_5.h5")

# predict...

weights1=reconstructed_model_1.predict(preprocessed_tensor)

# predict again...

# bridge=[]
# bridge.append(input_tensor_prepare(weights1[0,:,:,0]*255.))
# bridge=tf.cast(bridge,dtype=np.float32)

# weights1=reconstructed_model_1.predict(bridge)

# normalise...

# weigths1_max=np.max(weights1[0,:,:,2])
# weigths1_min=np.min(weights1[0,:,:,2])
# weigths1_mean=np.mean(weights1[0,:,:,2])
# norm=(weights1[0,:,:,2]-weigths1_min)/(weigths1_max-weigths1_min)#-((weigths1_mean-weigths1_min)/(weigths1_max-weigths1_min))
mod1=np.asanyarray(weights1*255,dtype=np.int8)
# avg=np.mean(mod1)
# threshold globally (not very good, some data are lost)

# masked=(mod1>=0.98)

# plot the results

# fig, axs = plt.subplots(nrows=1, ncols=2)
# axs[0].set_title('Processed JK - probability matrix')
# axs[0].imshow(mod1[0][:][:][:],cmap='twilight')
plt.imshow(mod1[0][:][:][:])
plt.show()
# axs[1].set_title('Raw JK - 8 bit BMP')
# axs[1].imshow(input_image_im,cmap='twilight')
# plt.imshow(input_image_im,cmap='twilight')
# plt.show()
# axs[2].set_title('Masked JK')
# axs[2].imshow(255*masked,cmap='twilight')
# plt.show()
