import keras
from keras.layers import Dropout, Activation, Lambda, PReLU, LeakyReLU, Conv2D, Add, add, Conv2DTranspose, Conv3D, Conv3DTranspose, BatchNormalization
import tensorflow as tf
import keras.backend as K


def residual_block_3D(x, filters, kernel_size=(3, 3, 3), depth=3, kernel_initializer='he_normal'):
    x = Conv3D(filters, kernel_size, activation='linear', padding='same')(x)
    shortcut = x
    
    for i in range(0, depth):
        x = Conv3D(filters, kernel_size, activation='linear', padding='same')(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
    x = add([shortcut, x])
    x = LeakyReLU()(x)

    return x

def down_conv_3D(x, filters, kernel_size=(2, 2, 2), strides=(2, 2, 2), kernel_initializer='he_normal'):
    # Please add activation layer when building the model.
    x = Conv3D(filters, kernel_size, strides=strides, padding='valid', activation='linear', kernel_initializer=kernel_initializer)(x)
    return x

def up_conv_3D(x , filters, kernel_size=(2, 2, 2), strides=None, kernel_initializer='he_normal'):
    # Please add activation layer when building the model.
    if strides is None:
        strides = kernel_size
    x = Conv3DTranspose(filters, kernel_size=kernel_size, strides=strides, activation='linear', kernel_initializer=kernel_initializer)(x)
    return x


#Remo's

def ActivationOp(layer_in, activation_type, name=None, l=0.1, shared_axes=(1, 2, 3)):
    if (activation_type != 'prelu') & (activation_type != 'leakyrelu'):
        return Activation(activation_type, name=name)(layer_in)
    elif activation_type == 'prelu':
        return PReLU(alpha_initializer=keras.initializers.Constant(value=l), shared_axes=shared_axes, name=name)(
            layer_in)
    else:
        # TODO: check if alpha should be 0.01 instead
        return LeakyReLU(l)(layer_in)

#   3D    
    
def ResidualBlock3D(layer_in, depth=3, kernel_size=5, filters=None, bn=None, dropout=0., activation='relu', kernel_initializer='he_normal',
                    name=None, dropout_layer=None, training=False):
    # Checking if we use BatchNorm
    if bn is not None:
        if bn == 'last':
            bn = 1
        elif bn == 'all':
            bn = 2
        else:
            raise NotImplementedError('"bn" has to be one of [None, "last", "all"]')
    else:
        bn = 0
#droplayer = Dropout(dropout, name='{}_drop'.format(name)) if dropout_layer is None else dropout_layer

    def drop(l):
        if dropout > 0. or not dropout_layer is None:
            return droplayer(l, training=training)
        else:
            return l
    # creates a residual block with a given depth for 3D input
    # there is NO non-linearity applied to the output! Has to be added manually
    layer_in = drop(layer_in)
    l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
               name='{}_c0'.format(name))(layer_in)

    if bn == 2 | (bn == 1 & depth == 1):
        l = BatchNormalization(name='{}_bn0'.format(name))(l)

    for i in range(1, depth):
        a = ActivationOp(l, activation, name='{}_a{}'.format(name, i - 1))
        a = drop(a)
        l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
                   name='{}_c{}'.format(name, i))(a)
        if bn == 2 | (bn == 1 & i == (depth-1)):
            l = BatchNormalization(name='{}_bn{}'.format(name, i))(l)
    o = Add()([layer_in, l])
    # o = Activation_wrap(o, activation, name='{}_a{}'.format(name,depth))
    return o

def ConvBlock3D(layer_in, depth=3, kernel_size=5, filters=None, bn=None, dropout=0., activation='relu', kernel_initializer='he_normal',
                    name=None, dropout_layer=None, training=False):
    # Checking if we use BatchNorm
    if bn is not None:
        if bn == 'last':
            bn = 1
        elif bn == 'all':
            bn = 2
        else:
            raise NotImplementedError('"bn" has to be one of [None, "last", "all"]')
    else:
        bn = 0

    droplayer = Dropout(dropout, name='{}_drop'.format(name)) if dropout_layer is None else dropout_layer

    def drop(l):
        if dropout > 0. or not dropout_layer is None:
            return droplayer(l, training=training)
        else:
            return l
        
    # creates a block of subsequent convolutions with a given depth for 3D input
    # there is NO non-linearity applied to the output! Has to be added manually
    layer_in = drop(layer_in)
    l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
               name='{}_c0'.format(name))(layer_in)

    if bn == 2 | (bn == 1 & depth == 1):
        l = BatchNormalization(name='{}_bn0'.format(name))(l)

    for i in range(1, depth):
        a = ActivationOp(l, activation, name='{}_a{}'.format(name, i - 1))
        a = drop(a)
        l = Conv3D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
                   name='{}_c{}'.format(name, i))(a)
        if bn == 2 | (bn == 1 & i == (depth-1)):
            l = BatchNormalization(name='{}_bn{}'.format(name, i))(l)
    o = l
    # o = Add()([layer_in, l])
    # o = Activation_wrap(o, activation, name='{}_a{}'.format(name,depth))
    return o


def DownConv3D(layer_in, kernel_size=2, strides=(2, 2, 2), filters=None, dropout=0., activation='relu',
               kernel_initializer='he_normal', name=None, dropout_layer=None, training=False):
    if isinstance(strides, int):
        strides = (strides, strides, strides)

    if not dropout_layer is None:
        layer_in = dropout_layer(layer_in, training=training)
    elif dropout > 0.:
        layer_in = Dropout(layer_in, name='{}_drop'.format(name))(layer_in, training=training)

    dc = Conv3D(filters, kernel_size, strides=strides, padding='valid', activation='linear',
                name='{}_dc0'.format(name), kernel_initializer=kernel_initializer)(layer_in)
    
    dc = ActivationOp(dc, activation, name='{}_a0'.format(name))
    return dc


def UpConv3D(layer_in, kernel_size=(2, 2, 2), strides=None, filters=None, dropout=0., activation='relu',
             kernel_initializer='he_normal', name=None, dropout_layer=None, training=False):
    if strides is None:
        strides = kernel_size
    elif isinstance(strides, int):
        strides = (strides, strides, strides)

    if not dropout_layer is None:
        layer_in = dropout_layer(layer_in, training=training)
    elif dropout > 0.:
        layer_in = Dropout(layer_in, name='{}_drop'.format(name))(layer_in)

    uc = Conv3DTranspose(filters, kernel_size=kernel_size, strides=strides, activation='linear',
                         name='{}_uc0'.format(name), kernel_initializer=kernel_initializer)(layer_in)
    uc = ActivationOp(uc, activation, name='{}_a0'.format(name))
    return uc


#   2D

def ResidualBlock2D(layer_in, depth=3, kernel_size=5, filters=None, bn=None, dropout=0., activation='relu', kernel_initializer='he_normal',
                    name=None, dropout_layer=None, training=False):
    # creates a residual block with a given depth for 2D input
    # there is NO non-linearity applied to the output! Has to be added manually

    droplayer = Dropout(dropout, name='{}_drop'.format(name)) if dropout_layer is None else dropout_layer
    def drop(l):
        if dropout > 0. or not dropout_layer is None:
            return droplayer(l, training=training)
        else:
            return l

    if bn is not None:
        if bn == 'last':
            bn = 1
        elif bn == 'all':
            bn = 2
        else:
            raise NotImplementedError('"bn" has to be one of [None, "last", "all"]')
    else:
        bn = 0

    layer_in = drop(layer_in)

    l = Conv2D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
               name='{}_c0'.format(name))(layer_in)

    if bn == 2 or ((bn == 1) and (depth == 1)):
        l = BatchNormalization(name='{}_bn0'.format(name))(l)

    for i in range(1, depth):
        a = ActivationOp(l, activation, name='{}_a{}'.format(name, i - 1))
        a = drop(a)
        l = Conv2D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
                   name='{}_c{}'.format(name, i))(a)
        if bn == 2 or ((bn == 1) and (i == (depth-1))):
            l = BatchNormalization(name='{}_bn{}'.format(name, i))(l)
    o = Add(name='{}_add'.format(name))([layer_in, l])
    return o

def ConvBlock2D(layer_in, depth=2, kernel_size=3, filters=None, dropout=0., activation='relu', kernel_initializer='he_normal',
                    name=None, dropout_layer=None, training=False):
    # creates a "convolution block" (series of regular convolutions) with a given depth for 2D input
    i = 0

    droplayer = Dropout(dropout, name='{}_drop'.format(name)) if dropout_layer is None else dropout_layer
    def drop(l):
        if dropout > 0. or not dropout_layer is None:
            return droplayer(l, training=training)
        else:
            return l

    layer_in = drop(layer_in)

    l = Conv2D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
               name='{}_c0'.format(name))(layer_in)

    for i in range(1, depth):
        a = ActivationOp(l, activation, name='{}_a{}'.format(name, i - 1))
        a = drop(a)
        l = Conv2D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer,
                   name='{}_c{}'.format(name, i))(a)
    o = ActivationOp(l, activation, name='{}_a{}'.format(name, i))
    return o

def DownConv2D(layer_in, kernel_size=2, strides=(2, 2), filters=None, dropout=0., activation='relu', kernel_initializer='he_normal',
               name=None, dropout_layer=None, training=False):

    if isinstance(strides, int):
        strides = (strides, strides)

    if not dropout_layer is None:
        layer_in = dropout_layer(layer_in, training=training)
    elif dropout > 0.:
        layer_in = Dropout(layer_in, name='{}_drop'.format(name))(layer_in)

    dc = Conv2D(filters, kernel_size, strides=strides, padding='valid', activation='linear',
                kernel_initializer=kernel_initializer,
                name='{}_dc0'.format(name))(layer_in)

    dc = ActivationOp(dc, activation, name='{}_a0'.format(name))
    return dc


def UpConv2D(layer_in, kernel_size=(2, 2), strides=None, filters=None, dropout=0., activation='relu',
             kernel_initializer='he_normal', name=None, dropout_layer=None, training=False):

    if strides is None:
        strides = kernel_size
    elif isinstance(strides, int):
        strides = (strides, strides)

    if not dropout_layer is None:
        layer_in = dropout_layer(layer_in, training=training)
    elif dropout > 0.:
        layer_in = Dropout(layer_in, name='{}_drop'.format(name))(layer_in)

    uc = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, activation='linear',
                         kernel_initializer=kernel_initializer,
                         name='{}_uc0'.format(name))(layer_in)
    uc = ActivationOp(uc, activation, name='{}_a0'.format(name))
    return uc

def UpScaleConv2D(layer_in, scale_factor=(2.,2.), kernel_size=None, strides=(1,1), filters=None, dropout=0., activation='relu', name=None, dropout_layer=None, training=False):
    # https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects
    # https://github.com/keras-team/keras/issues/5088#issuecomment-273851273
    def wrap_tf_resize_nearest_neighbor(x, size):
        import tensorflow as tf
        return tf.image.resize_nearest_neighbor(x, size=size)
    in_shape= K.int_shape(layer_in)
    s1, s2 = int(scale_factor[0]*in_shape[1]), int(scale_factor[1]*in_shape[2])
    up = Lambda(lambda x, scale1, scale2 : wrap_tf_resize_nearest_neighbor(x, size=K.constant([scale1, scale2], dtype='int32')), arguments={'scale1':s1,'scale2':s2}, name='{}_us0'.format(name))(layer_in)
    if not dropout_layer is None:
        layer_in = dropout_layer(layer_in, training=training)
    elif dropout > 0.:
        layer_in = Dropout(layer_in, name='{}_drop'.format(name))(layer_in, training=training)
    uc = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='linear',
                name='{}_c0'.format(name))(up)
    uc = ActivationOp(uc, activation, name='{}_a0'.format(name))

    return uc