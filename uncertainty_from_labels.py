import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

class uncertainty_from_labels():
    ## make the model give logits
    ## input_vector must be a tf tensor with dimension [batch_size, img_dim, img_dim, num_channel]
    ## make scalar_list of float type
    def __init__(self, model, scalar_list, batch_size, im_dim, im_channels):
        self.model = model
        self.scalar_list = tf.convert_to_tensor(scalar_list, dtype=float)
        self.batch_size = batch_size
        self.num_aug = len(self.scalar_list)
        self.im_dim = im_dim
        self.im_channels = im_channels
       
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

    def multi(self, mat1, mat2):
        i = 0
        mat1 = tf.expand_dims(mat1, axis = 0)
        while i < tf.shape(mat2)[0]:
            mat1 = tf.concat((mat1, tf.expand_dims(mat1[0]*tf.get_static_value(mat2[i]), axis = 0)), axis = 0)
            i = i + 1
        mat1 = tf.slice(mat1, [1, 0, 0, 0], [self.num_aug, self.im_dim, self.im_dim, self.im_channels])
        return mat1
        
        
    def generate_augmented_vectors(self, input_vector):
        old_vector = input_vector
        jacob = self.get_jacobian(input_vector)
        j = 0
        while j < self.batch_size:
            old_vector = tf.concat([old_vector, tf.expand_dims(input_vector[j], axis = 0) + self.multi(jacob[j], self.scalar_list)], axis = 0)
            j = j + 1
        gen_set = tf.slice(old_vector, [self.batch_size, 0, 0, 0], [(self.num_aug)*self.batch_size, self.im_dim, self.im_dim, self.im_channels])
        return gen_set

    def get_uncertainty(self, input_vector):
        a_vectors = self.generate_augmented_vectors(input_vector)
        waste1, y = self.get_top1_class(a_vectors)
        unc = []
        i = 0
        while i < self.batch_size:
            x = tf.slice(y, [i*self.num_aug], [self.num_aug])
            waste2, waste3, count = tf.unique_with_counts(x)
            count_max = tf.math.argmax(count)
            unc.append(tf.get_static_value(1 - (count[count_max])/self.num_aug))
            i = i + 1   
        return unc

    def show_aug_vectors(self, input_vector, i):
        waste, y = self.get_top1_class(input_vector)
        aug_vectors = self.generate_augmented_vectors(input_vector)
        waste2, z = self.get_top1_class(aug_vectors)
        unc = self.get_uncertainty(input_vector)
        x = tf.slice(aug_vectors, [i*self.num_aug, 0, 0, 0], [self.num_aug, self.im_dim, self.im_dim, self.im_channels])
        t = tf.slice(z, [i*self.num_aug], [self.num_aug])
        k = 0
        while k < self.num_aug:
            print(f'uncertainty = {unc[i]}')
            print(f'label = {tf.get_static_value(t[k])}')
            plt.imshow(x[k])
            plt.show()
            k = k + 1
                    

if __name__ == "__main__":
    print("file_done")