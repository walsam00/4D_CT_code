#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py

#Data import
enumeratorr = 0
buffer_size = 300
batch_size = 40
train_length = 350
epochs = 4001
steps_per_epoch = train_length // batch_size
save_pred_out, save_mask_out, save_rec_out = np.empty([16,240,240,1]), np.empty([16,240,240,1]), np.empty([16,240,240,1])
data_dir = '/folder/' #training and validation data directory - REPLACE WITH ACUTAL DIRECTORY
save_location = '/folder/' #directory for saving trained U-Net state - REPLACE WITH ACUTAL DIRECTORY

#called as last step in image loading
def normalize(rec_stack_in, seg_stack_in):
    rec_stack_out = tf.cast(rec_stack_in, tf.float32) #/ 255
    rec_stack_out = tf.image.per_image_standardization(rec_stack_out)
    seg_stack_in = seg_stack_in - 1
    return rec_stack_out, seg_stack_in

def load_image_training(rec_stack, seg_stack):
    rec_stack_out, seg_stack_out = normalize(rec_stack, seg_stack)
    return rec_stack_out, seg_stack_out

def load_image_validation(rec_stack, seg_stack):
    rec_stack_out, seg_stack_out = normalize(rec_stack, seg_stack)
    return rec_stack_out, seg_stack_out

#read images from disk
#training- and validation data is stored in four files each: two contain reconstructed CT image volumes, two contain the corresponding ground truth segmented data
#train_recon stores the reconstructed image data, train_seg stores the corresponding ground truth segmented data
train_recon = np.zeros([70,81,243,243,1])
train_seg = np.ones([70,81,243,243,1])

with h5py.File(data_dir + 'train_rec.h5','r') as hdf:
    d = hdf['data']
    train_recon[0:30,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'training_set_2/train_rec.h5', 'r') as hdf:
    d = hdf['data']
    train_recon[30:70,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'train_seg.h5', 'r') as hdf:
    d = hdf['data']
    train_seg[0:30,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'training_set_2/train_seg.h5', 'r') as hdf:
    d = hdf['data']
    train_seg[30:70,:,:,:,:] = np.array(d[...])

train_recon_cropped = np.zeros([350,16,240,240,1])
train_seg_cropped = np.zeros([350,16,240,240,1])

#training data needs to be cropped slightly and batched in the z-dimension to match input dimensions of U-Net
conversion_iterator = 0
while conversion_iterator < 70:
    train_recon_cropped[(conversion_iterator*5),:,:,:,:] = train_recon[conversion_iterator,0:16,2:-1,2:-1,:]
    train_recon_cropped[(conversion_iterator*5+1),:,:,:,:] = train_recon[conversion_iterator,16:32,2:-1,2:-1,:]
    train_recon_cropped[(conversion_iterator*5+2),:,:,:,:] = train_recon[conversion_iterator,32:48,2:-1,2:-1,:]
    train_recon_cropped[(conversion_iterator*5+3),:,:,:,:] = train_recon[conversion_iterator,48:64,2:-1,2:-1,:]
    train_recon_cropped[(conversion_iterator*5+4),:,:,:,:] = train_recon[conversion_iterator,64:80,2:-1,2:-1,:]

    train_seg_cropped[(conversion_iterator*5),:,:,:,:] = train_seg[conversion_iterator,0:16,2:-1,2:-1,:]
    train_seg_cropped[(conversion_iterator*5+1),:,:,:,:] = train_seg[conversion_iterator,16:32,2:-1,2:-1,:]
    train_seg_cropped[(conversion_iterator*5+2),:,:,:,:] = train_seg[conversion_iterator,32:48,2:-1,2:-1,:]
    train_seg_cropped[(conversion_iterator*5+3),:,:,:,:] = train_seg[conversion_iterator,48:64,2:-1,2:-1,:]
    train_seg_cropped[(conversion_iterator*5+4),:,:,:,:] = train_seg[conversion_iterator,64:80,2:-1,2:-1,:]
    conversion_iterator = conversion_iterator + 1

#convert to tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_recon_cropped, train_seg_cropped))

print('Created training dataset tensor')

#same for validation data: cropping and batching is necessary
val_proj = np.zeros([23,81,243,243,1])
val_seg = np.zeros([23,81,243,243,1])

with h5py.File(data_dir + 'val_rec.h5', 'r') as hdf:
    d = hdf['data']
    val_recon[0:10,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'training_set_2/val_rec.h5', 'r') as hdf:
    d = hdf['data']
    val_recon[10:23,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'val_seg.h5', 'r') as hdf:
    d = hdf['data']
    val_seg[0:10,:,:,:,:] = np.array(d[...])

with h5py.File(data_dir + 'training_set_2/val_seg.h5', 'r') as hdf:
    d = hdf['data']
    val_seg[10:23,:,:,:,:] = np.array(d[...])

val_proj_cropped = np.zeros([115,16,240,240,1])
val_seg_cropped = np.zeros([115,16,240,240,1])

conversion_iterator = 0
while conversion_iterator < 23:
    val_proj_cropped[(conversion_iterator*5),:,:,:,:] = val_recon[conversion_iterator,0:16,2:-1,2:-1,:]
    val_proj_cropped[(conversion_iterator*5+1),:,:,:,:] = val_recon[conversion_iterator,16:32,2:-1,2:-1,:]
    val_proj_cropped[(conversion_iterator*5+2),:,:,:,:] = val_recon[conversion_iterator,32:48,2:-1,2:-1,:]
    val_proj_cropped[(conversion_iterator*5+3),:,:,:,:] = val_recon[conversion_iterator,48:64,2:-1,2:-1,:]
    val_proj_cropped[(conversion_iterator*5+4),:,:,:,:] = val_recon[conversion_iterator,64:80,2:-1,2:-1,:]

    val_seg_cropped[(conversion_iterator*5),:,:,:,:] = val_seg[conversion_iterator,0:16,2:-1,2:-1,:]
    val_seg_cropped[(conversion_iterator*5+1),:,:,:,:] = val_seg[conversion_iterator,16:32,2:-1,2:-1,:]
    val_seg_cropped[(conversion_iterator*5+2),:,:,:,:] = val_seg[conversion_iterator,32:48,2:-1,2:-1,:]
    val_seg_cropped[(conversion_iterator*5+3),:,:,:,:] = val_seg[conversion_iterator,48:64,2:-1,2:-1,:]
    val_seg_cropped[(conversion_iterator*5+4),:,:,:,:] = val_seg[conversion_iterator,64:80,2:-1,2:-1,:]
    conversion_iterator = conversion_iterator + 1

#convert to tensorflow dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_proj_cropped, val_seg_cropped))
print('Created validation dataset tensor')

#tensor image processing
train = train_dataset.map(load_image_training, num_parallel_calls=tf.data.AUTOTUNE)
val = val_dataset.map(load_image_validation, num_parallel_calls=tf.data.AUTOTUNE)

#tensor batch shuffling
train_dataset = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val.batch(batch_size)

#Model Definition
#take random sample stack to show during training epochs
for image, mask in train_dataset.take(1):
    sample_image_stack, sample_mask_stack = image, mask

#number of segmentation layers
segmentation_classes = 4

#define downsample block
def downsample3D(filters, size, apply_batchnorm=True, pool_sizes=(2,2,2)):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3D(filters, size, strides=1, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.AveragePooling3D(pool_size=pool_sizes, padding='same'))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

#define upsample block
def upsample3D(filters, size, apply_dropout=False, pool_sizes=(2,2,2)):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3DTranspose(filters, size, strides=pool_sizes, padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result


#define network structure
def Generator():
    inputs = tf.keras.layers.Input(shape=[16,240,240,1])

    down_stack = [
        downsample3D(8, 3, pool_sizes=(1,1,1)),
        downsample3D(16, 3),                        #16,240,240,1 ->  8,120,120,16
        downsample3D(32, 3),                        #8,120,120,16 ->  4,60,60,32
        downsample3D(64, 3),                        #4,60,60,32   ->  2,30,30,64
        downsample3D(128, 3),                       #2,30,30,64  ->  1,15,15,128
        downsample3D(256, 3, pool_sizes=(1,3,3)),   #1,15,15,128 -> 1,5,5,256
    ]

    up_stack = [
        upsample3D(128, 2, pool_sizes=(1,3,3)),     #1,5,5,256      ->  1,15,15,128
        upsample3D(64, 2),                          #1,15,15,128    ->  2,30,30,64
        upsample3D(32, 2),                          #2,30,30,64     ->  4,60,60,32
        upsample3D(16, 2),                          #4,60,60,32     ->  8,120,120
        upsample3D(8, 2),                           #8,120,120      ->  16,240,240
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv3DTranspose(segmentation_classes, 5, strides=(1,1,1), padding='same')

    x = inputs

    #downsampling (make actual connections between layers)
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    #upsampling (make actual connections, perform skip concatonation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#Model Training
model = Generator()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', 'CategoricalAccuracy'])
#early stop condition: no improvement in validation_loss of more than 0.001 for over 200 epochs, best weights are restored once patience (200 epochs) runs out
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, verbose=1,mode='auto', baseline=None, restore_best_weights=True)

#actual Training
model_history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=None, validation_data=val_dataset, callbacks=[stop_callback])
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(EPOCHS)

print('loss')
print(loss)
print('validation loss')
print(val_loss)
print('whole history: ')
print(model_history.history)

print(model.count_params())
print(model.summary())

model.save(save_location + 'saved_model')