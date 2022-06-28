import tensorflow as tf
from uncertainty_from_distance import uncertainty_from_distances
from uncertainty_from_labels import uncertainty_from_labels

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32), label

def attack_fn(victim, stolen, dataloader, num_classes, im_dim, im_channels, batch_size, border_attack, ds_test):
  label = tf.random.uniform([1, num_classes])
  input = tf.random.uniform([1, im_dim, im_dim, im_channels])
  if border_attack == True:
    distance_obj = uncertainty_from_distances(victim, batch_size, 20, im_dim, im_channels)
  i = 0
  for (egs, wasted_label) in dataloader:
    current_batch = egs
    if border_attack == True:
      current_batch = distance_obj.generate_augmented_vectors(current_batch)      
    label = tf.concat([label, victim(current_batch)], axis = 0)
    input = tf.concat([input, current_batch], axis = 0)
    print(i)
    i = i + 1
  label = tf.slice(label, [1, 0], [tf.shape(label)[0] - 1, num_classes])
  input = tf.slice(input, [1, 0, 0, 0], [tf.shape(input)[0] - 1, im_dim, im_dim, im_channels])
  attack_data = tf.data.Dataset.from_tensor_slices((input, label))
  attack_data = attack_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  attack_data = attack_data.cache()
  attack_data = attack_data.batch(batch_size)
  attack_data = attack_data.prefetch(tf.data.AUTOTUNE) 
  stolen.compile(
  optimizer=tf.keras.optimizers.Adam(0.0003),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.CategoricalAccuracy()],
  )
  one_hot = tf.zeros([1, num_classes], dtype = tf.int64) 
  test_samples = tf.zeros([1, im_dim, im_dim, im_channels])
  for (test_eg, index) in ds_test:
    one_hot = tf.concat([one_hot, tf.one_hot(index, depth = num_classes, dtype = tf.int64)], axis = 0)
    test_samples = tf.concat([test_samples, test_eg], axis = 0)
  one_hot = tf.slice(one_hot, [1, 0], [tf.shape(one_hot)[0] - 1, num_classes]) 
  print(tf.shape(one_hot))
  # tf.print(one_hot, summarize = -1)
  test_samples = tf.slice(test_samples, [1, 0, 0, 0], [tf.shape(test_samples)[0] - 1, im_dim, im_dim, im_channels]) 
  print(tf.shape(test_samples))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_samples, one_hot))
  test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  test_dataset = test_dataset.cache()
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE) 
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_lr=0.00001)

  stolen.fit(attack_data, epochs = 50, validation_data = test_dataset, callbacks=[reduce_lr])

  return stolen