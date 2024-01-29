# use ImageJ for generaitng a testset for this network.
# i'll check if it is working.
# reshape (in a proper, logical way) will be added
# after training - just make 2'nd life for your old family photos!

import tensorflow as tf
import keras
import numpy as np
from PIL import Image
# from keras.utils.vis_utils import plot_model
import os, glob
# from keras.preprocessing.image import ImageDataGenerator
# something=(0,0,0) # just a tuple, for your work!
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
with tf.device('/device:GPU:0'):
    def last_4chars(x):
        return(x[-4:])
    raw_data_dir=f"/home/geniusz/nn/hires/dataset_hires/original/"
    # input_data_dir=f"D:\hires\dataset_hires\scaled"
    extension1="*.jpg"
    extension2="*.png"
    TF_ENABLE_ONEDNN_OPTS=0
    # dir=os.gir(raw_data_dir)
    val_dir = os.chdir(raw_data_dir)
    val_dir = os.listdir(raw_data_dir)
    val=[]
    for filename in sorted(val_dir, key = last_4chars):
        image=Image.open(f"/home/geniusz/nn/hires/dataset_hires/original/"+filename)
        image=np.asanyarray(image,dtype=np.float16)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        # print(np.shape(image))
        val.append(image)
    print(1)

    # for el in val:
    #     print(np.shape(el))

    inp_dir=os.chdir(raw_data_dir)
    inp_dir = os.listdir(raw_data_dir)

    inp=[]
    for filename in sorted(inp_dir, key = last_4chars):
        image=Image.open(f"/home/geniusz/nn/hires/dataset_hires/original/"+filename)
        image=image.resize((int(round(image.width/2)),int(round(image.height/2))), Image.Resampling.LANCZOS)
        image=np.asanyarray(image,dtype=np.float16)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        inp.append(image)
    # inp=np.asarray(inp,dtype=np.float16)

    # print(4.5)
    # for el in range(len(inp)):
    #     print(np.shape(inp[el]),np.shape(val[el]))
    # # print(inp.shape)
    # # inp=tf.convert_to_tensor(inp)  
    # print(len(val),len(inp))
    # datagen = ImageDataGenerator(
    #     rotation_range=350,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True, vertical_flip=True,zoom_range=2.5,brightness_range=(0.0,60))
    # ++++++++++++++++++ upscale 3x ...
    def superRes(n,batch_sizes):
        inputs = keras.layers.Input(shape=(None,None,3),batch_size=batch_sizes)#None,None

        x=keras.layers.Conv2D(3,(1,1),(1,1),activation='elu',padding='same')(inputs)
        y=keras.layers.Conv2D(3,(1,1),(1,1),activation='elu',padding='same')(inputs)
        z=keras.layers.Conv2D(3,(1,1),(1,1),activation='elu',padding='same')(inputs)
        a=keras.layers.Conv2D(3,(1,1),(1,1),activation='elu',padding='same')(inputs)

        bridge=keras.layers.Add()([z,a,x,y])

        sum1,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(bridge)

        x=keras.layers.Conv2D(3*n,(1,1),(1,1),activation='elu',padding='same')(sum1)
        y=keras.layers.Conv2D(3*n,(1,1),(1,1),activation='elu',padding='same')(sum1)
        z=keras.layers.Conv2D(3*n,(1,1),(1,1),activation='elu',padding='same')(sum1)
        a=keras.layers.Conv2D(3*n,(1,1),(1,1),activation='elu',padding='same')(sum1)

        sum2=keras.layers.Add()([y,y,z,a])
        sum2,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum2)

        x=keras.layers.Conv2DTranspose(n,(1,1),(2,2),activation="elu",padding="same")(sum2)
        y=keras.layers.Conv2DTranspose(n,(1,1),(2,2),activation="elu",padding="same")(sum2)
        z=keras.layers.Conv2DTranspose(n,(1,1),(2,2),activation="elu",padding="same")(sum2)
        a=keras.layers.Conv2DTranspose(n,(1,1),(2,2),activation="elu",padding="same")(sum2)


        sum2=keras.layers.Add()([y,y,z,a])
        sum2,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum2)

        x=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum2)
        y=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum2)
        z=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum2)
        a=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum2)

        sum3=keras.layers.Add()([x,y,z,a])
        sum3,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum3)

        x=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        y=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        z=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        a=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)

        sum3=keras.layers.Add()([x,y,z,a])
        sum3,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum3)

        x=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        y=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        z=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        a=keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)

        sum_final=keras.layers.Add()([x,y,z,a])
        sum_final,asdfsadf,gfggg=keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum_final)
        sumx=keras.layers.Conv2D(3,(1,1),(1,1),padding="same",activation="softmax")(sum_final)


        final=keras.Model(inputs,sumx)
        return final



    model=superRes(n=6,batch_sizes=1)
    # model.compile()
    model.summary()
    opti=keras.optimizers.Adam()

    epochs = 999

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for i in range(len(inp)):
            print(f"started {i}'th image...")

            # Convert NumPy arrays to TensorFlow tensors
            inputt = np.expand_dims(inp[i], axis=0)
            outt = np.expand_dims(val[i], axis=0)
            
            # Train on batch
            logits = model(inputt, outt)
            loss_value=keras.losses.binary_crossentropy(outt,logits)
            loss_value2=np.linalg.norm(loss_value)
            print("loss: " + str(loss_value2))
            # print("accuracy: " + str(accuracy))

        # Save the model after each epoch
        model.save("model_epoch_{}.h5".format(epoch))