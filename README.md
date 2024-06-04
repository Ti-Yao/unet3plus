### Example usage in tensorflow

```python 
from unet3plus import *
outputsize = 2 # number of output channels

input_shape = [128,128,1]
output_shape = [128,128,outputsize]

inputs = tf.keras.Input(shape = input_shape)
unet3 = unet3plus(inputs, 
                  rank = 2,  # dimension
                  n_outputs = outputsize, 
                  add_dropout = 1, # 1 or 0 to add dropout
                  dropout_rate = 0.3,
                  base_filters = 32, 
                  kernel_size = 3, 
                  stack_num_down =  2, 
                  stack_num_up = 1, 
                  supervision = 1) # 1 or 0 to add supervision

model = tf.keras.Model(inputs = inputs, outputs = unet3.outputs())

```
