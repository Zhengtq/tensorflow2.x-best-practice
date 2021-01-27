#!/usr/bin/env python
#coding: utf-8


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import data_util as du
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import datetime
import  create_vit_classifier


GLOBAL_BATCH_SIZE = 128
num_classes = 1
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = GLOBAL_BATCH_SIZE
ALL_SAMPLES = float(2594870)
checkpoint_PATH = './checkpoint/'

try:
    os.makedirs(checkpoint_PATH)
except:
    pass




physical_devices = tf.config.list_physical_devices('GPU')
for ind, item in enumerate(physical_devices):
    tf.config.experimental.set_memory_growth(item, True)


#  TRAIN_GPUS = [0,1,2,3]
#  devices = ["/gpu:{}".format(i) for i in TRAIN_GPUS]
#  strategy = tf.distribute.MirroredStrategy(devices)

tf.config.set_visible_devices(physical_devices[0:8], 'GPU') 
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


train_ds =  du.load_dataset1(batch_size = GLOBAL_BATCH_SIZE)
test_ds =  du.load_dataset_test(batch_size = GLOBAL_BATCH_SIZE)

options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
train_ds = train_ds.with_options(options)

train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
test_dist_dataset = strategy.experimental_distribute_dataset(test_ds)




with strategy.scope():
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()
#    @tf.function
    def compute_acc(labels, predictions, now_trace):
        predictions = tf.squeeze(predictions)
        predictions = tf.nn.sigmoid(predictions)
        now_trace.update_state(labels, predictions)
        return 0


with strategy.scope():
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

#    @tf.function
    def compute_loss(labels, predictions):
        predictions = tf.squeeze(predictions)
        labels = tf.cast(labels, tf.float32)
        per_example_loss = loss_object(labels, predictions)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = predictions)
        per_example_loss = tf.reduce_sum(per_example_loss)/float(GLOBAL_BATCH_SIZE)
        return per_example_loss


with strategy.scope():
    model = create_vit_classifier()
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
  #  optimizer = tfa.optimizers.SGDW(learning_rate=0.01, weight_decay=weight_decay, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)





with strategy.scope():
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        compute_acc(labels, predictions, train_accuracy)
        return loss

    def test_step(inputs):
        images, labels = inputs
        predictions = model(images, training=False)
        compute_acc(labels, predictions, test_accuracy)



def run_experiment():

    with strategy.scope():
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

    epoc_step = ALL_SAMPLES/GLOBAL_BATCH_SIZE
    start_time = datetime.datetime.now()
    for t_step, x in enumerate(train_dist_dataset):

        if t_step == 500:
              tf.profiler.experimental.start('/tmp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if t_step == 600:
              tf.profiler.experimental.stop()


        with tf.profiler.experimental.Trace('Train', step_num=t_step, _r=1):
            step_loss = distributed_train_step(x)

        if t_step % 100 == 0:
            step_loss = step_loss.numpy()
            train_acc = train_accuracy.result().numpy()

            end_time = datetime.datetime.now()
            time_interval = (end_time - start_time).seconds
            start_time = datetime.datetime.now()
            epoc_time = time_interval * epoc_step / 100.0 / 3600
            now_epoc = float(t_step) /epoc_step

            print("now_epoc: " + fm(now_epoc), \
                "train_acc: " +  fm(train_acc, 4),\
                "train_loss: " + fm(step_loss),\
                "epoc_time: " + str(epoc_time)[:4],\
                "now_step: " + str(t_step))

      
        if t_step % 1000 == 0:
            test_accuracy.reset_states()
            for x_test in test_dist_dataset:
                distributed_test_step(x_test)
            test_acc = test_accuracy.result().numpy()
            print('test_acc: '+ fm(test_acc, 4))
           
            if test_acc > 0.85 or t_step % 10000 == 0:
                model.save_weights(checkpoint_PATH + str(test_acc)[:4] + '_' + str(t_step) + '.ckpt')


        t_step += 1



    return 0


history = run_experiment()

