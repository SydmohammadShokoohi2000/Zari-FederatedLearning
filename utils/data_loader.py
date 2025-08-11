import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import h5py

logger = logging.getLogger("FL-DataLoader")

def load_and_prepare_car_hacking_data_federated(
    processed_data_dir, num_clients, client_ids, batch_size
):
    logger.info(f"Loading pre-processed data from HDF5 file in: {processed_data_dir}")
    
    hdf5_path = os.path.join(processed_data_dir, "processed_data.h5")

    if not os.path.exists(hdf5_path):
        error_msg = f"Pre-processed HDF5 file not found: {hdf5_path}. Please run the preprocess_data.py script first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    hf = h5py.File(hdf5_path, 'r')
    all_samples = hf['samples']
    all_labels = hf['labels']
    logger.info(f"Loaded {len(all_samples)} samples from HDF5 file.")
    
    unique_labels, counts = np.unique(all_labels[:], return_counts=True) 
    num_classes_actual = len(unique_labels)
    logger.info(f"Data has {num_classes_actual} unique classes: {dict(zip(unique_labels, counts))}")

    try:
        train_indices, test_indices = train_test_split(
            np.arange(len(all_samples)), test_size=0.2, random_state=42, stratify=all_labels[:]
        )
    except ValueError:
        train_indices, test_indices = train_test_split(
            np.arange(len(all_samples)), test_size=0.2, random_state=42
        )

    test_data_global_dl = None; test_data_num_global = len(test_indices)
    if test_data_num_global > 0:
        test_data_global_dl = tf.data.Dataset.from_tensor_slices(
            (all_samples[sorted(test_indices)], all_labels[sorted(test_indices)])
        ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_data_num_total = len(train_indices)
    train_data_local_num_map = {}; train_data_local_map_dl = {}
    test_data_local_map_dl = {cid: None for cid in client_ids}

    if num_clients > 0 and train_data_num_total > 0:
        np.random.shuffle(train_indices)
        client_idx_splits = np.array_split(train_indices, num_clients)
        for i, client_id in enumerate(client_ids):
            client_idx = sorted(client_idx_splits[i]) 
            train_data_local_num_map[client_id] = len(client_idx)
            if len(client_idx) > 0:
                train_data_local_map_dl[client_id] = tf.data.Dataset.from_tensor_slices(
                    (all_samples[client_idx], all_labels[client_idx])
                ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            else:
                train_data_local_map_dl[client_id] = tf.data.Dataset.from_tensor_slices((np.array([]), np.array([]))).batch(batch_size)
    
    return [
        train_data_num_total, test_data_num_global, None, test_data_global_dl,
        train_data_local_num_map, train_data_local_map_dl, test_data_local_map_dl, num_classes_actual
    ]