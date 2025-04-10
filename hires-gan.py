#  hires gan
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

with tf.device('CPU:0'):
    def last_4chars(x):
        return x[-4:]


    raw_data_dir = r"/media/tk/9C96B25096B22B22/hires/dataset_hires/original"

    extension1="*.jpg"
    extension2="*.png"
    TF_ENABLE_ONEDNN_OPTS=0
    # dir=os.gir(raw_data_dir)
    val_dir = os.chdir(raw_data_dir)
    val_dir = os.listdir(raw_data_dir)
    raw_images=[]
    for filename in sorted(val_dir, key = last_4chars):
        image=Image.open(raw_data_dir+f"/"+filename)
        image=np.asanyarray(image,dtype=np.float32)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        # print(np.shape(image))
        raw_images.append(image)
    print(1)

    # for el in val:
    #     print(np.shape(el))

    inp_dir=os.chdir(raw_data_dir)
    inp_dir = os.listdir(raw_data_dir)

    scaled_images=[]
    for filename in sorted(inp_dir, key = last_4chars):
        image=Image.open(raw_data_dir+f"/"+filename)
        image=image.resize((int(round(image.width//2)),int(round(image.height//2))), Image.Resampling.LANCZOS)
        image=np.asanyarray(image,dtype=np.float32)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        scaled_images.append(image)
    def superResg(n,batch_sizes):
        inputs = keras.layers.Input(shape=(None,None,3),batch_size=batch_sizes)#None,None

        x=tf.keras.layers.Conv2DTranspose(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        y=tf.keras.layers.Conv2DTranspose(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        z=tf.keras.layers.Conv2DTranspose(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        a=tf.keras.layers.Conv2DTranspose(n,(1,1),(4,4),activation="elu",padding="same")(inputs)

        x=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(y)
        y=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(x)
        z=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(z)
        a=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(a)

        sum3=tf.keras.layers.Add()([x,y,z,a])
        # sum3,asdfsadf,gfggg=tf.keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum3)

        x=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)
        y=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(a)
        z=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(x)
        a=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(z)

        sum3=tf.keras.layers.Add()([x,y,z,a])

        x=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(x)
        y=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(y)
        z=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(z)
        a=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum3)

        sum_final=tf.keras.layers.Add()([x,y,z,a])
        sumx=tf.keras.layers.Conv2D(3,(1,1),(1,1),padding="same",activation="softmax")(sum_final)
        # print(tf.shape(sumx))


        final=tf.keras.Model(inputs,sumx)
        return final

    def superResd(n,batch_sizes):
        inputs = keras.layers.Input(shape=(None,None,3),batch_size=batch_sizes)#None,None


        x=tf.keras.layers.Conv2D(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        y=tf.keras.layers.Conv2D(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        z=tf.keras.layers.Conv2D(n,(1,1),(4,4),activation="elu",padding="same")(inputs)
        a=tf.keras.layers.Conv2D(n,(1,1),(4,4),activation="elu",padding="same")(inputs)


        sum2=tf.keras.layers.Add()([y,y,z,a])
    
        x=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(a)
        y=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(x)
        z=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(y)
        a=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(sum2)

        x=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(x)
        y=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(y)
        z=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(z)
        a=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu")(a)

        sum_final=tf.keras.layers.Add()([x,y,z,a])
        # sum_final,asdfsadf,gfggg=tf.keras.layers.ConvLSTM1D(n,(1),return_state=True,return_sequences=True)(sum_final)
        sumx=tf.keras.layers.Conv2D(3,(1,1),(1,1),padding="same",activation="softmax")(sum_final)
        # print(tf.shape(sumx))
        flatten = tf.keras.layers.Lambda(lambda x: tf.compat.v1.reduce_mean(x, axis=[1, 2]))(sumx)

        # sumd=tf.keras.layers.Dense(1,"sigmoid")(flatten)

        final=tf.keras.Model(inputs,flatten)
        return final
    print(2)
    model_generator=superResg(n=2,batch_sizes=1)
    model_discriminator=superResd(n=2,batch_sizes=1)
    model_generator.compile()
    model_discriminator.compile()
    generator_optimizer = tf.keras.optimizers.AdamW(1e-4)
    discriminator_optimizer = tf.keras.optimizers.AdamW(1e-4)

    epochs = 999



    def generator_loss(fake_output):
        """Generator loss for GAN."""
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(real_output, fake_output):
        """Discriminator loss for GAN."""
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss



    @tf.function
    def train_step(lres,hres, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = model_generator(lres, training=True)
            real_output = model_discriminator(hres, training=True)
            fake_output = model_discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_gen = gen_tape.gradient(gen_loss, model_generator.trainable_variables)
        gradients_disc = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_gen, model_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_disc, model_discriminator.trainable_variables))

        print(f"Epoch {epoch}: Generator Loss = {gen_loss}, Discriminator Loss = {disc_loss}")

    epochs=10000
    print('\n\n...main loop...\n\n')
    for epoch in range(epochs):
        for i in range(len(scaled_images[0])):
            # print(np.shape(scaled_images[i]),np.shape(np.stack((scaled_images[i],)*1,axis=0)))
            train_step(np.stack((scaled_images[i],)*1,axis=0),np.stack((raw_images[i],)*1,axis=0),epoch)
