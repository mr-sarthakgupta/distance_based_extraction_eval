import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

class uncertainty_from_distances():
    ## make the model give softmax
    ## input_vector must be a tf tensor with dimension [batch_size, img_dim, img_dim, num_channel]
    ## make scalar_list of float type
    def __init__(self, model, batch_size, binary_limit, img_dim, img_channels):
        self.model = model
        self.batch_size = batch_size
        self.binary_limit = binary_limit
        self.img_dim = img_dim
        self.img_channels = img_channels
       
    def get_soft_out(self, input_vector):
        return self.model(input_vector)

    def get_top1_class(self, input_vector):
        out = self.get_soft_out(input_vector)
        top1_index = tf.math.argmax(out ,axis = 1)
        top1_list = tf.math.reduce_max(out, axis = 1)
        return top1_list, top1_index

    def get_jacobian(self, input_vector):
        with tf.GradientTape(persistent=False) as g:
            g.watch(input_vector)
            y, waste = self.get_top1_class(input_vector)
            y = tf.reshape(y, [self.batch_size, 1])
        jacob = tf.squeeze(g.batch_jacobian(y, input_vector), axis = 1)
        return jacob

    def generate_augmented_vectors(self, input_vector):
        aug_vector = input_vector
        jacob = self.get_jacobian(input_vector)
        waste, og_out = self.get_top1_class(input_vector)
        aug_out = og_out
        flag_cnt = np.zeros(self.batch_size)
        i = 0
        while np.sum(flag_cnt) < self.batch_size and i < 35:
            k = 0
            while k < self.batch_size:
                if aug_out[k] == og_out[k]:
                    aug_vector = tf.concat([aug_vector, tf.expand_dims(input_vector[k] - jacob[k]*(10**i), axis = 0)], axis = 0)
                if aug_out[k] != og_out[k]:
                    flag_cnt[k] = 1
                    aug_vector = tf.concat([aug_vector, tf.expand_dims(aug_vector[k], axis = 0)], axis = 0)
                k = k + 1
            aug_vector = tf.slice(aug_vector, [self.batch_size, 0, 0, 0], [self.batch_size, self.img_dim, self.img_dim, self.img_channels])
            waste1, aug_out = self.get_top1_class(aug_vector)
            i = i + 1
        return aug_vector

    def binary_search(self, input_vector):
        sample1 = tf.Variable(input_vector, dtype=float)
        waste, y_og = self.get_top1_class(input_vector)
        sample2 = tf.Variable(self.generate_augmented_vectors(input_vector), dtype=float)
        i = 0
        while i < self.binary_limit:
            mid_sample = (sample1 + sample2)/2
            waste, y = self.get_top1_class(mid_sample)
            j = 0
            while j <self.batch_size:
                if y[j] == y_og[j]:
                    sample1[j].assign(mid_sample[j])
                else:
                    sample2[j].assign(mid_sample[j])
                j = j + 1
            i = i + 1
        final_vector = mid_sample 
        return final_vector

    def get_uncertainty(self, input_vector):
        return tf.math.reduce_euclidean_norm(self.binary_search(input_vector) - input_vector, keepdims = False, axis = (1, 2, 3))

    def show_aug_vectors(self, input_vector):
        waste, y = self.get_top1_class(input_vector)
        aug_vectors = self.binary_search(input_vector)
        waste2, z = self.get_top1_class(aug_vectors)
        unc = self.get_uncertainty(input_vector)
        i = 0
        while i < self.batch_size:
            print(f'uncertainty of sample = {unc[i]}')
            print(f'label of input vector = {y[i]}')
            plt.imshow(input_vector[i])
            plt.show()
            print(f'label of input vector = {z[i]}')
            plt.imshow(aug_vectors[i])
            plt.show()
            print('################################################################################################################')
            i = i + 1
                    

if __name__ == "__main__":
    print("file_done")  