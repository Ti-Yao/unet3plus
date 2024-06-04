import tensorflow as tf
import layer_util 
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Layer, Add, Activation
from tensorflow.keras.layers import Input, Conv3D, SpatialDropout3D, UpSampling3D, MaxPooling3D, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Concatenate, Average, LeakyReLU, BatchNormalization
def conv_res_block(inputs, num_filters):
    # First convolutional layer
    x1 = Conv3D(num_filters, kernel_size = 3, padding='same',kernel_initializer = 'he_normal')(inputs)
    x1 = InstanceNormalization(axis = -1)(x1)
    
    # Second convolutional layer
    x = Conv3D(num_filters, kernel_size = 3, padding='same',kernel_initializer = 'he_normal')(inputs)
    x = InstanceNormalization(axis = -1)(x)
    x = Activation('LeakyReLU')(x)
    x = SpatialDropout3D(rate=0.3)(x) 

    x = Conv3D(num_filters, kernel_size = 3, padding='same',kernel_initializer = 'he_normal')(x)
    x = InstanceNormalization(axis = -1)(x)
    
    x = Add()([x1, x])
    x = Activation('LeakyReLU')(x)
    
    return x


class unet3plus:
    def __init__(self, 
                 inputs, 
                 rank = 2, 
                 n_outputs = 3, 
                 add_dropout = True,
                 dropout_rate = 0.3,
                 base_filters = 32, 
                 kernel_size = 3, 
                 stack_num_down = 2, 
                 stack_num_up = 1, 
                 batch_norm = True, 
                 supervision= True):
        
        
        self.inputs = inputs
        self.rank = rank
        self.n_outputs = n_outputs
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.add_dropout = add_dropout
        self.dropout_rate = dropout_rate        
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up 
        self.batch_norm = batch_norm
        self.supervision = supervision
        self.conv_config = dict(kernel_size = 3,
                           padding = 'same',
                           kernel_initializer = 'he_normal')
    
    def aggregate(self, scale_list, name):
        X = tf.keras.layers.concatenate(scale_list, axis = -1)
        X = self.conv_block(X, self.base_filters * 5, num_stacks = self.stack_num_up)
        return X

    def deep_sup(self, inputs, scale):
        size = 2 ** (scale-1)
        conv = layer_util.get_nd_layer('Conv', self.rank)
        upsamp = layer_util.get_nd_layer('UpSampling', self.rank)
        if self.rank == 2:
            upsamp_config = dict(size=(size , size), interpolation='bilinear')
        else:
            upsamp_config = dict(size=(size , size, size))
            
            
        X = inputs  
        X = conv(self.n_outputs, activation = None, **self.conv_config, name = f'deepsup_conv_{scale}')(X)
        if scale != 1:
            X = upsamp(**upsamp_config, name = f'deepsup_upsamp_{scale}')(X)
        X = tf.keras.layers.Activation(activation = 'sigmoid' if self.n_outputs == 1 else 'softmax', name = f'deepsup_activation_{scale}')(X)
        return X
        
        
    def full_scale(self, inputs, to_layer, from_layer, stack_num_up):
        layer_diff = from_layer - to_layer  
        size = 2 ** (abs(layer_diff))
        conv = layer_util.get_nd_layer('Conv', self.rank)
        maxpool = layer_util.get_nd_layer('MaxPool', self.rank)
        upsamp = layer_util.get_nd_layer('UpSampling', self.rank)
        if self.rank == 2:
            upsamp_config = dict(size=(size , size), interpolation='bilinear')
        else:
            upsamp_config = dict(size=(size , size, size))
        
        X = inputs        
        if to_layer < from_layer:
            X = upsamp(**upsamp_config, name = f'fullscale_{from_layer}_{to_layer}')(X)
        elif to_layer > from_layer:
            X = maxpool(pool_size=(size , size) if self.rank == 2 else (size,size, size), name = f'fullscale_maxpool_{from_layer}_{to_layer}')(X)
        X = self.conv_block(X, self.base_filters, num_stacks = stack_num_up)
        return X
        
    def conv_block(self, inputs, filters, num_stacks):
        conv = layer_util.get_nd_layer('Conv', self.rank)
        
        X = inputs
        for i in range(num_stacks):
            X = conv(filters, **self.conv_config)(X)
            if self.batch_norm:
                X = tf.keras.layers.BatchNormalization(axis=-1)(X)
            X = tf.keras.layers.ReLU()(X)
            
        return X
    
    def encode(self, inputs, scale, num_stacks):
        maxpool = layer_util.get_nd_layer('MaxPool', self.rank)
        filters = self.base_filters * 2 ** scale
        
        X = inputs
        X = self.conv_block(X, filters, num_stacks)
        if scale >= 4 and self.add_dropout:
            X = tf.keras.layers.Dropout(rate = self.dropout_rate, name = f'encoding_{scale}_dropout')(X)
        if scale != 5:
            X = maxpool(pool_size=(2 , 2) if self.rank == 2 else (2,2,2), name = f'encoding_{scale}_maxpool')(X)
        return X
        
    def outputs(self):
        conv = layer_util.get_nd_layer('Conv', rank = self.rank)

        X = self.inputs
        XE1 = self.encode(X, scale = 1, num_stacks = self.stack_num_down)
        XE2 = self.encode(XE1, scale = 2, num_stacks = self.stack_num_down)
        XE3 = self.encode(XE2, scale = 3, num_stacks = self.stack_num_down)
        XE4 = self.encode(XE3, scale = 4, num_stacks = self.stack_num_down)
        XE5 = self.encode(XE4, scale = 5, num_stacks = self.stack_num_down)

        XD4_from_XE5 = self.full_scale(XE5, 4, 5, self.stack_num_up)
        XD4_from_XE4 = self.full_scale(XE4, 4, 4, self.stack_num_up)
        XD4_from_XE3 = self.full_scale(XE3, 4, 3, self.stack_num_up)
        XD4_from_XE2 = self.full_scale(XE2, 4, 2, self.stack_num_up)
        XD4_from_XE1 = self.full_scale(XE1, 4, 1, self.stack_num_up)
        XD4 = self.aggregate([XD4_from_XE5, XD4_from_XE4, XD4_from_XE3, XD4_from_XE2, XD4_from_XE1], name = 'agg_XD4')
        
        XD3_from_XE5 = self.full_scale(XE5, 3, 5, self.stack_num_up)
        XD3_from_XD4 = self.full_scale(XD4, 3, 4, self.stack_num_up)
        XD3_from_XE3 = self.full_scale(XE3, 3, 3, self.stack_num_up)
        XD3_from_XE2 = self.full_scale(XE2, 3, 2, self.stack_num_up)
        XD3_from_XE1 = self.full_scale(XE1, 3, 1, self.stack_num_up)
        XD3 = self.aggregate([XD3_from_XE5, XD3_from_XD4, XD3_from_XE3, XD3_from_XE2, XD3_from_XE1], name = 'agg_XD3')
        
        XD2_from_XE5 = self.full_scale(XE5, 2, 5, self.stack_num_up)
        XD2_from_XE4 = self.full_scale(XE4, 2, 4, self.stack_num_up)
        XD2_from_XD3 = self.full_scale(XD3, 2, 3, self.stack_num_up)
        XD2_from_XE2 = self.full_scale(XE2, 2, 2, self.stack_num_up)
        XD2_from_XE1 = self.full_scale(XE1, 2, 1, self.stack_num_up)
        XD2 = self.aggregate([XD2_from_XE5, XD2_from_XE4, XD2_from_XD3, XD2_from_XE2, XD2_from_XE1], name = 'agg_XD2')
        
        XD1_from_XE5 = self.full_scale(XE5, 1, 5, self.stack_num_up)
        XD1_from_XE4 = self.full_scale(XE4, 1, 4, self.stack_num_up)
        XD1_from_XE3 = self.full_scale(XE3, 1, 3, self.stack_num_up)
        XD1_from_XD2 = self.full_scale(XD2, 1, 2, self.stack_num_up)
        XD1_from_XE1 = self.full_scale(XE1, 1, 1, self.stack_num_up)
        XD1 = self.aggregate([XD1_from_XE5, XD1_from_XE4, XD1_from_XE3, XD1_from_XD2, XD1_from_XE1], name = 'agg_XD1')
        
        
        if self.supervision == True:
            XE5 = self.deep_sup(XE5, 5)
            XD4 = self.deep_sup(XD4, 4)
            XD3 = self.deep_sup(XD3, 3)
            XD2 = self.deep_sup(XD2, 2)
        XD1 = self.deep_sup(XD1, 1)
        
        
        if self.supervision == True:
            return [XE5,XD4,XD3,XD2,XD1]
        else:
            return XD1