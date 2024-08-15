import argparse
import os
from sklearn.model_selection import KFold
from typing import List, Tuple
import numpy as np
import SimpleITK as sitk
from shutil import copyfile
import time

from umcglib.augment import augment, random_crop
from umcglib.utils import print_stats_np, read_yaml_to_dict, set_gpu, print_
from umcglib.images import znorm_n
from umcglib.losses import categorical_focal_loss

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, Activation, add
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU, Dense, multiply
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, Permute, Reshape
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.models import load_model


################################  README  ######################################
# NEW - This script will loaded in RSS Reconstructions data as train data 
# and segmentations as label data.
# We will run a segmentation model and hyper parameters from Chris his research.


################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for splitting training, validation and the test set.')

    parser.add_argument(
        '-c',
        '--config_fpath',
        type=str,
        required=True,
        help='Path to conig file with hyper parameters for a training run.',
    )

    args = parser.parse_args()

    return args


class IntermediateImages(Callback):
    def __init__(self,
        validation_set,
        prefix, 
        num_images=10
    ):
        self.num_images = min(validation_set[0].shape[0], num_images)
        self.prefix = prefix
        self.validation_set = (
            validation_set[0][:self.num_images, ...],
            validation_set[1][:self.num_images, ...]
        )
        
        # Export scan crops and targets once
        # they don't change during training so we export them only once
        for img_idx in range(self.num_images):
            for ci in range(self.validation_set[0].shape[-1]):
                img_s = sitk.GetImageFromArray(self.validation_set[0][img_idx, :, :, :, ci].squeeze().T)
                sitk.WriteImage(img_s, f"{prefix}_{img_idx:03d}_coil{ci}_img.nii.gz")
            seg_s = sitk.GetImageFromArray(
                        np.argmax(self.validation_set[1][img_idx], axis=-1).squeeze().astype(np.float32).T)
            sitk.WriteImage(seg_s, f"{prefix}_{img_idx:03d}_seg.nii.gz")


    def on_epoch_end(self, epoch, logs={}):
        # Predict on the validation_set
        predictions = self.model.predict(self.validation_set, batch_size=1)
        
        for i in range(self.num_images):
            prd_s = sitk.GetImageFromArray(
                np.argmax(predictions[i], axis=-1).astype(np.float32).squeeze().T)
            sitk.WriteImage(prd_s, f"{self.prefix}_{i:03d}_pred.nii.gz")
        
class PrintLearningRate(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        learningrate = K.eval(self.model.optimizer._decayed_lr(tf.float64))
        print(f"\nLearning rate at epoch {epoch} is {learningrate}")
        print('learning rate method 2', self.model.optimizer.get_config())        
        print('learning rate method 3', K.eval(self.model.optimizer.lr))


def categorical_dice_coefficient(class_idx, name):
    def dice_coef(y_true, y_pred):
        yt_argmax = K.argmax(y_true)
        yp_argmax = K.argmax(y_pred)
        yt_class = K.cast(K.equal(yt_argmax, class_idx), K.floatx())
        yp_class = K.cast(K.equal(yp_argmax, class_idx), K.floatx())
        intersection = K.sum(yt_class * yp_class)
        return (2. * intersection + K.epsilon()) / (
            K.sum(yt_class) + K.sum(yp_class) + K.epsilon())
    metric = dice_coef
    metric.__name__ = name
    return metric


def mean_dice_np(y_true, y_pred, num_classes):
    dices = []
    for c in range(1, num_classes):
        y_class_true = (y_true == c) * 1.
        y_class_pred = (y_pred == c) * 1.

        nom = np.sum(y_class_pred * y_class_true)
        den = np.sum(y_class_pred) + np.sum(y_class_true)

        dc = (2 * nom + 1e-7) / (den + 1e-7)
        dices.append(dc)
    return sum(dices) / len(dices)


def squeeze_excite_block(input, ratio=8):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x


def build_dual_attention_unet(
    input_shape,
    num_classes,
    l2_regularization = 0.0001,
    instance_norm = False
    ):

    def conv_layer(x, kernel_size, out_filters, strides=(1,1,1)):
        x = Conv3D(out_filters, kernel_size, 
                strides             = strides,
                padding             = 'same',
                kernel_regularizer  = l2(l2_regularization),
                kernel_initializer  = 'he_normal',
                use_bias            = False
                )(x)
        return x
    
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    def conv_block(input, out_filters, strides=(1,1,1), with_residual=False, with_se=False, activation='relu'):
        # Strided convolution to convsample
        x = conv_layer(input, (3,3,3), out_filters, strides)
        x = Activation('relu')(x)

        # Unstrided convolution
        x = conv_layer(x, (3,3,3), out_filters)

        # Add a squeeze-excite block
        if with_se:
            se = squeeze_excite_block(x)
            x = add([x, se])
            
        # Add a residual connection using a 1x1x1 convolution with strides
        if with_residual:
            residual = conv_layer(input, (1,1,1), out_filters, strides)
            x = add([x, residual])
            
        if instance_norm:
            x = InstanceNormalization(**in_defaults)(x)
            
        if activation == 'leaky':
            x = LeakyReLU(alpha=.1)(x)
        else:
            x = Activation('relu')(x)
        
        # Activate whatever comes out of this
        return x

    # If we already have only one input, no need to combine anything
    inputs = Input(input_shape)

    # Downsampling
    conv1 = conv_block(inputs, 16)
    conv2 = conv_block(conv1, 32, strides=(2,2,2), with_residual=True, with_se=True) #72x72x18
    conv3 = conv_block(conv2, 64, strides=(2,2,2), with_residual=True, with_se=True) #36x36x18
    conv4 = conv_block(conv3, 128, strides=(2,2,2), with_residual=True, with_se=True) #18x18x9
    conv5 = conv_block(conv4, 256, strides=(2,2,2), with_residual=True, with_se=True) #9x9x9
    
    # First upsampling sequence
    up1_1 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(conv5) #18x18x9
    up1_2 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_1) #36x36x18
    up1_3 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_2) #72x72x18
    bridge1 = concatenate([conv4, up1_1]) #18x18x9 (128+128=256)
    dec_conv_1 = conv_block(bridge1, 128, with_residual=True, with_se=True, activation='leaky') #18x18x9

    # Second upsampling sequence
    up2_1 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_1) # 36x36x18
    up2_2 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(up2_1) # 72x72x18
    bridge2 = concatenate([conv3, up1_2, up2_1]) # 36x36x18 (64+128+64=256)
    dec_conv_2 = conv_block(bridge2, 64, with_residual=True, with_se=True, activation='leaky')
    
    # Final upsampling sequence
    up3_1 = Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_2) # 72x72x18
    bridge3 = concatenate([conv2, up1_3, up2_2, up3_1]) # 72x72x18 (32+128+64+32=256)
    dec_conv_3 = conv_block(bridge3, 32, with_residual=True, with_se=True, activation='leaky')
    
    # Last upsampling to make heatmap
    up4_1 = Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_3) # 72x72x18
    dec_conv_4 = conv_block(up4_1, 16, with_residual=False, with_se=True, activation='leaky') #144x144x18 (16)

    # Reduce to a single output channel with a 1x1x1 convolution
    single_channel = Conv3D(num_classes, (1, 1, 1))(dec_conv_4)  

    # Apply sigmoid activation to get binary prediction per voxel
    act  = Activation('softmax')(single_channel)

    # Model definition
    model = Model(inputs=inputs, outputs=act)
    return model


def build_unet(
    window_size,
    num_classes, 
    l2_regularization = 0.0001, 
    instance_norm     = False
):
    # Default parameters for conv layers
    c_defaults = {
        "kernel_size" : (3,3,3),
        "kernel_initializer" : 'he_normal',
        "padding" : 'same'
    }
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    # Create NAMED input layers for each sequence
    ct_input  = Input(window_size)

    # Contraction path
    # he_normal defines initial weights - it is a truncated normal distribution (Gaussian dist.)
    # sets padding to same, meaning that input dimensions are the same as output dimensions
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(ct_input)
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c1)
    if instance_norm:
        c1 = InstanceNormalization(**in_defaults)(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(p1)
    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c2)
    if instance_norm:
        c2 = InstanceNormalization(**in_defaults)(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(p2)
    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c3)
    if instance_norm:
        c3 = InstanceNormalization(**in_defaults)(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(p3)
    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c4)
    if instance_norm:
        c4 = InstanceNormalization(**in_defaults)(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(p4)
    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(c5)
    if instance_norm:
        c5 = InstanceNormalization(**in_defaults)(c5)
    c5 = Activation('relu')(c5)

    # Upwards U part
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(u6)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c6)
    if instance_norm:
        c6 = InstanceNormalization(**in_defaults)(c6)
    c6 = Activation('relu')(c6)

    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(u7)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c7)
    if instance_norm:
        c7 = InstanceNormalization(**in_defaults)(c7)
    c7 = Activation('relu')(c7)

    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(u8)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c8)
    if instance_norm:
        c8 = InstanceNormalization(**in_defaults)(c8)
    c8 = Activation('relu')(c8)

    u9 = UpSampling3D((2, 2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(u9)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c9)
    if instance_norm:
        c9 = InstanceNormalization(**in_defaults)(c9)
    c9 = Activation('relu')(c9)

    # Perform 1x1x1 convolution and reduce the feature maps to N channels.
    output_layer = Conv3D(num_classes, (1, 1, 1), 
        padding='same', 
        activation='softmax'
        )(c9)

    unet = Model(
        inputs=ct_input,
        outputs=output_layer
        )

    return unet


def _dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)


def dice_coef_multilabel(y_true, y_pred, class_weights: dict , numLabels=7):
    dice=0

    for index in range(0, numLabels):
        dice -= class_weights[index] * _dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice


# Wrapper function so that the number of classes can be an extra argument.
def get_dice_coef_multilabel(class_weights:dict, n_classes=7):
    def dice_multilbl(y_true, y_pred):
        return dice_coef_multilabel(y_true, y_pred, class_weights=class_weights, numLabels=n_classes)
    return dice_multilbl


def write_np_coil2nifti_to_temp(input_n: np.ndarray, namething: str = "", coil_idx: int = 0) -> None:
    input_n = np.around(input_n*1000, decimals=0).astype(np.float32)
    print_stats_np(input_n, f"{namething}_coil{coil_idx}")
    input_s = sitk.GetImageFromArray(input_n.T)
    fname = os.path.join(TEMP_DIR, f"{namething}_coilidx{coil_idx}.nii.gz")
    sitk.WriteImage(input_s, fname)
    print_(f"\n>Wrote to: {fname}")


def get_generator_train(
    ksps_paths: List[str],
    seg_paths: List[str],
    indexes: List[int],
    batch_size: int = None,
    window_size: Tuple[int,int,int] = None,
    do_augmentation: bool = True,
    do_shuffle: bool = True,
    n_classes: int = 7,
    rotation_freq: float = 0.1,
    tilt_freq: float = 0.1,
    noise_freq: float = 0.3,
    noise_mult: float = 1e-3,
    mirror_freq: float = 0.5,
    norm: str = "div1000",
    seed: int = None,
    dtype = np.float32,
    sometitle: str = "",
    coil_imgs_arr_shape = (512, 512, 196, 18),
):
    batch_size = len(indexes) if batch_size == None else batch_size
    
    idx       = 0   #index taken from indexes
    batch_idx = 0   #simple counter of batches

    input_batch  = np.zeros((batch_size,) + window_size + (coil_imgs_arr_shape[-1],), dtype=dtype)
    output_batch = np.zeros((batch_size,) + window_size + (n_classes,), dtype=dtype)
    coil_imgs    = np.zeros(coil_imgs_arr_shape, dtype=np.float32)

    rng = np.random.default_rng(seed)
    
    while True:
        print_(f"\n>Prepping {sometitle} batch {batch_idx}, with pid {os.getpid()}...")
        t = time.time()

        if idx == 0 and do_shuffle:
            rng.shuffle(indexes)
        
        current_index = indexes[idx]

        for ci in range(coil_imgs_arr_shape[-1]):    #ci = coil_index
            coil_imgs[:,:,:,ci] = np.load(os.path.join(ksps_paths[current_index], f"coil_{ci}_kspace.npy")).astype(np.float32)
            if norm == "div1000":
                    coil_imgs[:,:,:,ci] = coil_imgs[:,:,:,ci] / 1000.
        seg_img = np.load(seg_paths[current_index])

        # if False:
        #     fname = os.path.join(TEMP_DIR, f"seg.nii.gz")
        #     sitk.WriteImage(sitk.GetImageFromArray(seg_img.astype(np.float32)), fname)
        #     print_(f"\n>Wrote to: {fname}")
        #     fname = os.path.join(TEMP_DIR, f"entire_coil.nii.gz")
        #     sitk.WriteImage(sitk.GetImageFromArray(np.around((coil_imgs[:,:,:,10]*1000).T, decimals=0).astype(np.float32).T), fname)
        #     print_(f"\n>Wrote to: {fname}")

        for crop_idx in range(batch_size):
            print_(f"\t>Prepping image crop: {crop_idx+1}/{batch_size}")

            # Preprocessing per coil
            for ci in range(coil_imgs.shape[-1]):
                # print_(f"\t\t>Prepping coil idx: {ci+1}/{coil_imgs.shape[-1]}")

                if ci == 0:
                    input_batch[crop_idx, :, :, :, ci], seg_img_crop = random_crop(
                        img               = coil_imgs[:,:,:,ci],
                        shape             = window_size,
                        seg               = seg_img,
                        oversample_center = True,
                        seed              = seed + crop_idx + batch_idx
                    )
                else:   # no segmentation crop needed here
                    input_batch[crop_idx, :, :, :, ci] = random_crop(
                        img               = coil_imgs[:,:,:,ci],
                        shape             = window_size,
                        seg               = None,
                        oversample_center = True,
                        seed              = seed + crop_idx + batch_idx,
                    )

                if do_augmentation:
                     input_batch[crop_idx, :, :, :, ci], seg_img_crop = augment(
                        img            = input_batch[crop_idx, :, :, :, ci],
                        seg            = seg_img_crop,
                        noise_chance   = noise_freq,
                        noise_mult_max = noise_mult,
                        rotate_chance  = rotation_freq,
                        tilt_chance    = tilt_freq,
                        mirror_chance  = mirror_freq,
                        seed           = seed + batch_idx,
                    )
                
                # if False:
                #     if ci < 2:
                #         write_np_coil2nifti_to_temp(np.squeeze(input_batch[crop_idx, :, :, :, ci]), f"{sometitle}_batch{batch_idx}_img{crop_idx}", ci)
        
            output_batch[crop_idx] = to_categorical(seg_img_crop, n_classes)

        # Increase the current index and modulo by the number of rows
        # so that we stay within bounds
        idx = (idx + 1) % len(indexes)


        print_(f"\t>Made {sometitle} input  batch with shape: {input_batch.shape} with dtype {input_batch.dtype}")     
        print_(f"\t>Made {sometitle} output batch with shape: {output_batch.shape} with dype {output_batch.dtype}")
        print_(f"\t>Done Prepping batch {batch_idx} with time: {round(time.time() -t, 1)}s.")
        
        batch_idx += 1
        yield input_batch, output_batch


def get_generator_val(
    ksps_paths: List[str],
    seg_paths: List[str],
    indexes: List[int],
    batch_size: int = None,
    window_size: Tuple[int,int,int] = None,
    do_augmentation: bool = True,
    do_shuffle: bool = True,
    n_classes: int = 7,
    rotation_freq: float = 0.1,
    tilt_freq: float = 0.1,
    noise_freq: float = 0.3,
    noise_mult: float = 1e-3,
    mirror_freq: float = 0.5,
    norm: str = "znorm",
    seed: int = None,
    dtype = np.float32,
    sometitle: str = "",
    coil_imgs_arr_shape = (512, 512, 196, 4),
):
    batch_size = len(indexes) if batch_size == None else batch_size
    
    idx       = 0   #index taken from indexes
    batch_idx = 0   #simple counter of batches

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + window_size + (coil_imgs_arr_shape[-1],), dtype=dtype)
    output_batch = np.zeros((batch_size,) + window_size + (n_classes,), dtype=dtype)
    coil_imgs = np.zeros(coil_imgs_arr_shape, dtype=np.float32)

    rng = np.random.default_rng(seed)
    
    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        print_(f"\n>Prepping {sometitle} batch {batch_idx}, with pid {os.getpid()}...")
        t = time.time()

        for img_idx in range(batch_size):
            print_(f"\t>Prepping image idx: {img_idx+1}/{batch_size}")

            if idx == 0 and do_shuffle:
                rng.shuffle(indexes)

            current_index = indexes[idx]

            for ci in range(coil_imgs_arr_shape[-1]):    #ci = coil_index
                coil_imgs[:,:,:,ci] = np.load(os.path.join(ksps_paths[current_index], f"coil_{ci}_kspace.npy")).astype(np.float32)
            
            # print_(f"Coil array shape after loading: {coil_imgs.shape}")
            # Data loading
            seg_img = np.load(seg_paths[current_index])

            # Preprocessing per coil
            for ci in range(coil_imgs.shape[-1]):
                
                if norm == "div1000":
                    coil_imgs[:,:,:,ci] = coil_imgs[:,:,:,ci] / 1000.

                # crop
                if ci == 0:
                    input_batch[img_idx, :, :, :, ci], seg_img = random_crop(
                        img   = coil_imgs[:,:,:,ci],
                        shape = window_size,
                        seg   = seg_img,
                        seed  = seed + img_idx + batch_idx
                    )
                else:   # no segmentation crop needed here
                    input_batch[img_idx, :, :, :, ci] = random_crop(
                        img   = coil_imgs[:,:,:,ci],
                        shape = window_size,
                        seg   = None,
                        seed  = seed + img_idx + batch_idx
                    )

                if do_augmentation:
                     input_batch[img_idx, :, :, :, ci], seg_img = augment(
                        img            = input_batch[img_idx, :, :, :, ci],
                        seg            = seg_img,
                        noise_chance   = noise_freq,
                        noise_mult_max = noise_mult,
                        rotate_chance  = rotation_freq,
                        tilt_chance    = tilt_freq,
                        mirror_chance  = mirror_freq,
                        seed           = seed + img_idx + batch_idx,
                    )
                
                if False:
                    write_np_coil2nifti_to_temp(np.squeeze(input_batch[img_idx, :, :, :, ci]), f"{sometitle}_batch{batch_idx}_img{img_idx}", ci)
        
            output_batch[img_idx] = to_categorical(seg_img, n_classes)

            # Increase the current index and modulo by the number of rows
            # so that we stay within bounds
            idx = (idx + 1) % len(indexes)

        print_(f"\t>Made {sometitle} input  batch with shape: {input_batch.shape} with dtype {input_batch.dtype}")     
        print_(f"\t>Made {sometitle} output batch with shape: {output_batch.shape} with dype {output_batch.dtype}")
        print_(f"\t>Done Prepping batch {batch_idx} with time: {round(time.time() -t, 1)}s.")
        
        batch_idx += 1
        yield input_batch, output_batch


def train(
    train_path_list: str,
    label_path_list: str,
    pretrained_model_dir: str,
    class_weights: dict,
    loss: str = "dice_coef_multilabel",
    optimizer: str = "rmsprop",
    learning_rate: float = 0.0004,
    window_size: List = (160, 160, 48),
    idxs_path: str = "data/path_lists/train_val_test_idxs_n300.yml",
    train_set_key: str = "train_set0",
    batch_size: int = 13,
    n_classes: int = 7,
    rotation_freq: float = 0.25,
    tilt_freq: float = 0.040,
    noise_freq: float = 0.60,
    noise_mult: float = 0.0030,
    mirror_freq: float = 0.5,
    seed: int = None,
    fold_num: int = 0,
    num_folds: int = 10,
    class_names = ["back", "fem_c", "tib_c", "pat_c", "fem", "tib", "pat"],
    unet_type: str = "simple",
    l2_regularization: float = 0.0001,
    do_instance_norm: bool = True,
    normalization: str = "znorm",
    do_early_stop: bool = True,
    early_stop_pat: int = 50,
    early_stop_var: str = "val_loss",
    early_stop_mode: str = "min",
    max_epochs = 5000,
    validation_interval = 10,
    coil_imgs_arr_shape: int = (512, 512, 196, 18),      # The shape of the multi-coild img array
    pretrained: bool = False,
    **kwargs,
):

    if DEBUG:
        coil_imgs_arr_shape = (512, 512, 196, 18)
        validation_interval = 2

    print_("> Fold number:", fold_num, "of", num_folds)

    # Get all the files that can be loaded
    ksp_dirs = [l.strip() for l in open(train_path_list)]
    seg_files = [l.strip() for l in open(label_path_list)]

    all_indexes = read_yaml_to_dict(idxs_path)[train_set_key]
    if DEBUG:
        all_indexes = all_indexes[:int(DEBUG_PERC_LOAD*len(all_indexes))]

    kfold = KFold(num_folds, shuffle=True, random_state=seed)
    train_idxs, valid_idxs = list(kfold.split(all_indexes))[fold_num]
    list(train_idxs)
    list(valid_idxs)

    print_("\n>Dataset division:")
    print_("\t>train indexes:", train_idxs, f" with length {len(train_idxs)}")
    print_("\t>Valid indexes:", valid_idxs, f" with length {len(valid_idxs)}")

    train_generator = get_generator_train(
        ksps_paths          = ksp_dirs,
        seg_paths           = seg_files,
        indexes             = train_idxs,
        batch_size          = batch_size if not DEBUG else 4,
        window_size         = window_size,
        do_augmentation     = True,
        do_shuffle          = True,
        n_classes           = n_classes,
        rotation_freq       = rotation_freq,
        tilt_freq           = tilt_freq,
        noise_freq          = noise_freq,
        noise_mult          = noise_mult,
        mirror_freq         = mirror_freq,
        norm                = normalization,
        seed                = seed,
        sometitle           = "train",
        coil_imgs_arr_shape  = coil_imgs_arr_shape,
    )

    val_generator = get_generator_val(
        ksps_paths          = ksp_dirs,
        seg_paths           = seg_files,
        indexes             = valid_idxs,
        batch_size          = 90 if not DEBUG else 1,
        window_size         = window_size,
        do_augmentation     = False,
        do_shuffle          = True,
        n_classes           = n_classes,
        norm                = normalization,
        seed                = seed,
        sometitle           = "val",
        coil_imgs_arr_shape  = coil_imgs_arr_shape,
    )
    # train_set = next(train_generator)
    valid_set = next(val_generator)

    # Add a dice coefficient for each non-background class
    dice_metrics = []
    for class_idx in range(1, n_classes):
        dice_metrics+=[categorical_dice_coefficient(class_idx, class_names[class_idx])] 

    # Add a mean dice score over all non-background classes
    def mean_dice(y_true, y_pred):
        return sum([m(y_true, y_pred) for m in dice_metrics])/len(dice_metrics)

    if optimizer == "rmsprop":
        optimizer = RMSprop(learning_rate)
    if loss == "dice_coef_multilabel":
        loss = get_dice_coef_multilabel(class_weights=class_weights, n_classes=n_classes)
    if loss == "categorical_focal_loss":
        loss = categorical_focal_loss(alpha=0.25)
    if loss == "categoricalcrossentropy":
        loss = CategoricalCrossentropy()

    # Create the model and show summary
    if pretrained:
        print(f'>> pre-trained model: {pretrained_model_dir} <<')
        class_weights = {0:1,1:2,2:2,3:2,4:1,5:1,6:1}
        dependencies = {
            'loss' : get_dice_coef_multilabel(class_weights=class_weights, n_classes=n_classes),
            'dice_multilbl' : get_dice_coef_multilabel(class_weights=class_weights, n_classes=n_classes),
            '0b': '0b',
            '1fc':'1fc', 
            '2tc':'2tc',
            '3pc':'3pc', 
            '4fe':'4fe',  
            '5ti':'5ti', 
            '6pa':'6pa',
            'mean_dice':'mean_dice'
            }
        dnn = load_model(
            filepath = pretrained_model_dir,
            custom_objects = dependencies,
            compile = False)
        dnn.compile(
            optimizer   = optimizer,
            loss        = loss,
            metrics     = dice_metrics + [mean_dice])

    else:
        print('>> fresh model <<')
        if unet_type == "simple":
            dnn = build_unet(
                window_size=window_size + (coil_imgs_arr_shape[-1],), 
                num_classes=n_classes,
                l2_regularization=l2_regularization,
                instance_norm=do_instance_norm)
        if unet_type == "dual_attention":
            dnn = build_dual_attention_unet(
                input_shape=window_size + (coil_imgs_arr_shape[-1],),
                num_classes=n_classes,
                l2_regularization=l2_regularization,
                instance_norm=do_instance_norm)
        dnn.summary(line_length=160)
        dnn.compile(
            optimizer   = optimizer,
            loss        = loss,
            metrics     = dice_metrics + [mean_dice])

    callbacks = []

    if do_early_stop:
        # Stop training after X epochs without improvement
        callbacks += [
            EarlyStopping(
                patience        = early_stop_pat,
                monitor         = early_stop_var,
                mode            = early_stop_mode,
                verbose         = 1
            )
        ]

    callbacks += [
        IntermediateImages(
        validation_set  = valid_set, 
        prefix          = os.path.join(OUTPUT_DIR, f"train-fold{fold_num}"), 
        num_images      = len(valid_set[0]))
    ]

    callbacks += [PrintLearningRate()]

    callbacks += [CSVLogger(os.path.join(LOG_DIR, f"train_fold{fold_num}.csv"))]

    for metric_name in [m.__name__ for m in dice_metrics] + ["loss", "mean_dice"]:
        callbacks += [
            ModelCheckpoint(
                os.path.join(MODEL_DIR, f"best_{metric_name}_fold{fold_num}.h5"),
                monitor         = f"val_{metric_name}", 
                save_best_only  = True, 
                mode            = 'min' if "loss" in metric_name else "max",
                verbose         = 1
            )
        ]

    # Train the model we created
    dnn.fit(train_generator,
        validation_data    = valid_set,
        steps_per_epoch    = len(train_idxs) // batch_size * validation_interval, 
        epochs             = max_epochs,
        callbacks          = callbacks,
        verbose            = 1,
    )

    print_("[I] Completed.")


def read_config(config_path, verbatim=True):
    
    # read
    configs = read_yaml_to_dict(config_path)
    
    # fix some params
    configs['window_size'] = tuple(configs['window_size'])
    configs['seed'] = None if not configs['use_seed'] else configs['seed']
    
    # print
    if verbatim:
        print_("\nConfigs")
        for key in configs:
            print_(f"{key} {type(configs[key])}:\t{configs[key]}")
        print_()
    return configs


################################################################################


if __name__ == '__main__':

    args = parse_input_args()
    configs = read_config(args.config_fpath)
    
    DEBUG           = configs['is_debug']
    VERBOSE         = configs['verbose'] if not DEBUG else 1
    DEBUG_PERC_LOAD = 0.08
    set_gpu(gpu_idx=0) if not DEBUG else set_gpu(gpu_idx=1)
    
    TRAIN_DIR = os.path.join(configs['train_dir'], f"fold{configs['fold_num']}")

    if DEBUG:
        TRAIN_DIR = os.path.join(TRAIN_DIR, 'debug')

    print_("> Creating folder structure")
    SEGMENTATION_DIR = os.path.join(TRAIN_DIR, "segmentations/")
    OUTPUT_DIR       = os.path.join(TRAIN_DIR, "output/")
    SAMPLE_DIR       = os.path.join(TRAIN_DIR, "samples/")
    LOG_DIR          = os.path.join(TRAIN_DIR, "logs/")
    MODEL_DIR        = os.path.join(TRAIN_DIR, "models/")
    UNDERSAMPLE_DIR  = os.path.join(TRAIN_DIR, "undersamples/")
    TEMP_DIR         = os.path.join(TRAIN_DIR, "temp/")
    FIGS_DIR         = os.path.join(TRAIN_DIR, "figs/")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UNDERSAMPLE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    dst = os.path.join(TRAIN_DIR, args.config_fpath.split('/')[-1])
    copyfile(args.config_fpath, dst)

    # for on peregrine - To ensure we do not go OOM (RAM)
    disable_eager_execution()
    
    train(**configs)
