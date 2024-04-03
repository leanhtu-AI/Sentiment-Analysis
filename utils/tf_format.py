import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

def to_tensorflow_format(tokenized_dataset):
    features = tokenized_dataset.features
    features_to_keep = list(features.keys())[:-3]  # Remove the last 3 features
    return tokenized_dataset.remove_columns(features_to_keep).with_format('tensorflow')

def preprocess_tokenized_dataset(tokenized_dataset, tokenizer, labels, batch_size, shuffle=False):
    tf_dataset = to_tensorflow_format(tokenized_dataset)
    features = {x: tf.convert_to_tensor(tf_dataset[x]) for x in tokenizer.model_input_names}
    labels = labels.reshape(len(labels), -1)
    tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(tf_dataset))
    return tf_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
