import argparse
import os
from pickletools import optimize
from unicodedata import name
import numpy as np
from tensorflow.python.eager.monitoring import Metric
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf


STAGE = "creating base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    
    # get the data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full/255.0
    X_test = X_test/255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    ## set the seeds
    seed = 2021 # we can also get seed from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define layers
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28]),
            tf.keras.layers.Dense(300, name="hidden_layer_1"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(100, name="hidden_layer_2"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
        ]

    ## define the model and compile it
    model = tf.keras.models.Sequential(LAYERS)

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3) # we can also just write "SGD" if we are fine with default learning rate
    METRICS = ["accuracy"]

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    model.summary()



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e