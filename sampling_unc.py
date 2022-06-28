from uncertainty_from_distance import uncertainty_from_distances
from uncertainty_from_labels import uncertainty_from_labels
import tensorflow as tf
import numpy as np

def sample_uniform(dataset, fraction, unc_obj, im_dim, im_channels):
    sampled_vec_list = []
    sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels], dtype = tf.float32)
    full_sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels])
    i = 0
    for (a, b) in dataset:
        ordered_unc_list = tf.make_ndarray(tf.make_tensor_proto(unc_obj.get_uncertainty(a)))
        argmax_list = np.argsort(ordered_unc_list)
        sampled_index = np.random.choice(argmax_list, size = int(len(argmax_list)*fraction))
        sampled_index = np.sort(sampled_index)
        full_sampled_vec = tf.concat([full_sampled_vec, tf.gather(params = a, indices = sampled_index)], axis = 0)
        print(i)
        i = i + 1
    full_sampled_vec = tf.slice(full_sampled_vec, [1, 0, 0, 0], [tf.shape(full_sampled_vec)[0] - 1, im_dim, im_dim, im_channels])
        
    return full_sampled_vec

def sample_high_unc(dataset, fraction, unc_obj, im_dim, im_channels):
    sampled_vec_list = []
    sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels], dtype = tf.float32)
    full_sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels])
    i = 0
    for (a, b) in dataset:
        ordered_unc_list = tf.make_ndarray(tf.make_tensor_proto(unc_obj.get_uncertainty(a)))
        argmax_list = np.argsort(ordered_unc_list)
        sampled_index = argmax_list[0:int(len(argmax_list)*fraction):1]
        sampled_index = np.sort(sampled_index)
        full_sampled_vec = tf.concat([full_sampled_vec, tf.gather(params = a, indices = sampled_index)], axis = 0)
        print(i)
        i = i + 1
    full_sampled_vec = tf.slice(full_sampled_vec, [1, 0, 0, 0], [tf.shape(full_sampled_vec)[0] - 1, im_dim, im_dim, im_channels])
        
    return full_sampled_vec

def sample_low_unc(dataset, fraction, unc_obj, im_dim, im_channels):
    sampled_vec_list = []
    sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels], dtype = tf.float32)
    full_sampled_vec = tf.zeros([1, im_dim, im_dim, im_channels])
    i = 0
    for (a, b) in dataset:
        ordered_unc_list = tf.make_ndarray(tf.make_tensor_proto(unc_obj.get_uncertainty(a)))
        argmax_list = np.argsort(ordered_unc_list)
        sampled_index = argmax_list[-int(len(argmax_list)*fraction):len(argmax_list):1]
        sampled_index = np.sort(sampled_index)
        full_sampled_vec = tf.concat([full_sampled_vec, tf.gather(params = a, indices = sampled_index)], axis = 0)
        print(i)
        i = i + 1
    full_sampled_vec = tf.slice(full_sampled_vec, [1, 0, 0, 0], [tf.shape(full_sampled_vec)[0] - 1, im_dim, im_dim, im_channels])
        
    return full_sampled_vec