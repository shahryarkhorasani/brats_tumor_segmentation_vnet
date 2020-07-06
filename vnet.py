from keras import Model
from keras.layers import Input, Dense, SpatialDropout3D, LeakyReLU, Add, Conv3D, Conv3DTranspose, Concatenate
from ops import *

def vnet(x,y,z, channels=4, n_classes=3, filters=[18,36,86,168,308], summary=False):
    input = Input((x,y,z,channels))
    # Downward
    layer = Conv3D(filters[0],kernel_size=(3, 3, 3), padding='same')(input)
    layer_0 = LeakyReLU()(layer)
    layer = down_conv_3D(layer_0,filters[1])
    layer = LeakyReLU()(layer)
    
    layer_1 = residual_block_3D(layer,filters[1],depth=3)
    layer = down_conv_3D(layer_1,filters[1])
    layer = LeakyReLU()(layer)
    
    layer_2 = residual_block_3D(layer,filters[2]) 
    layer = down_conv_3D(layer_2,filters[2])
    layer = LeakyReLU()(layer)
    
    layer_3 = residual_block_3D(layer,filters[3])
    layer = down_conv_3D(layer_3,filters[3])
    layer = LeakyReLU()(layer)
        
    #Bottom
    layer = residual_block_3D(layer,filters[4],depth=4)
    
    #Upward
    layer = up_conv_3D(layer, filters[3])
    layer = SpatialDropout3D(0.3)(layer)
    layer = LeakyReLU()(layer)
    layer = Concatenate()([layer, layer_3])
    layer = residual_block_3D(layer , filters[4])
    
    layer = up_conv_3D(layer, filters[2])
    layer = SpatialDropout3D(0.3)(layer)
    layer = LeakyReLU()(layer)
    layer = Concatenate()([layer, layer_2])
    layer = residual_block_3D(layer , filters[3])
    
    layer = up_conv_3D(layer, filters[1])
    layer = SpatialDropout3D(0.3)(layer)
    layer = LeakyReLU()(layer)
    layer = Concatenate()([layer, layer_1])
    layer = residual_block_3D(layer , filters[2], depth=3)
    
    layer = up_conv_3D(layer, filters[1])
    layer = LeakyReLU()(layer)
    layer = Concatenate()([layer, layer_0])
    layer = Conv3D(filters=n_classes, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(layer)
    
    model = Model([input],[layer])
    if summary:
        print(model.summary(line_length=140))
    return model
