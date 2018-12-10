from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import resnet
from . import keras_modules_injection


@keras_modules_injection
def ResNet(*args, **kwargs):
    return resnet.ResNet_custom(*args, **kwargs)
'''def ResNet(stack_args,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    def stack_fn(x):
        x = resnet_common.stack1(x, 64, stack_args[0], stride1=1, name='conv2')
        x = resnet_common.stack1(x, 128, stack_args[1], name='conv3')
        x = resnet_common.stack1(x, 256, stack_args[2], name='conv4')
        x = resnet_common.stack1(x, 512, stack_args[3], name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet50',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)'''

