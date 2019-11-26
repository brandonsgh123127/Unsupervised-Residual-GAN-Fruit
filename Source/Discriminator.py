from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.framework import ops
ops.reset_default_graph()



"""
# A discriminator allows to differentiate between what is real and what is fake.  For example,
# think of this as an 'officer' trying to look for a culprit, closely examining features.  By using 
# CNNs, this network is able to detect what is real and what is fake.
"""
class Discriminator(object):
    #Initialize image shape
    def __init__(self,image_shape):
        self.image_shape = image_shape

        # Creating a CNN using multiple layers....
    def define_discriminator(self):
        # initialize weight
        # source image
        g_init = RandomNormal(mean=1.0, stddev=0.2)
        init = RandomNormal(stddev=0.2)
        in_src_image = Input(shape=self.image_shape)
        print("descrim shape int ", self.image_shape)
        # target image
        #in_target_image = Input(shape=self.image_shape)
        # place both source and target images in one
       # merged_images = Concatenate()([in_src_image, in_target_image])
        """
        CONVOLUTIONAL NETWORK FOR DISCRIMINATOR!!!   
        """
        d = Conv2D(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(1024, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(d)
        d1 = BatchNormalization(gamma_initializer=g_init)(d)

        d = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d1)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Concatenate()([d, d1])
        d = LeakyReLU(alpha=0.2)(d)
        d = Flatten()(d)
        d = Dropout(0.2)(d)

        # Now, allow for only 1 filter at end of CNN
        d2 = Dense(1, kernel_initializer=init)(d)


        # define model
        model = Model(in_src_image,d2)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        #discriminator_model = Model(inputs=in_src_image, outputs=model)
        return model


