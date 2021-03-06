import argparse
from email.mime import base
import os
from pickletools import optimize
from tabnanny import verbose
from unicodedata import name
import numpy as np
from tensorboard import summary
from tensorflow.python.eager.monitoring import Metric
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import io


STAGE = "greater than 5 transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_greater_than_5_labels(list_of_labels):

    for idx, label in enumerate(list_of_labels):
        greater_than_5_condition = label > 5
        list_of_labels[idx] = np.where(greater_than_5_condition, 1, 0)

    return list_of_labels


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

    y_train_bin, y_test_bin, y_valid_bin = update_greater_than_5_labels([y_train, y_test, y_valid])

    ## set the seeds
    seed = 2021 # we can also get seed from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

   ## log our model summary information in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str


    ## load the base model
    base_model_path = os.path.join("artifacts","models","base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    ## model.summary()
    logging.info(f"{STAGE} model summary: \n{_log_model_summary(base_model)}")


    ## define layers
    # layers are already defined in the base model
    # here we eill have to freeze weights of previous layers and added new trainable layers over it

    ## freeze the weights
    for layer in base_model.layers[:-1]: # -1 beacuse we dont want output layer as we will add new output layer
        print(f"trainable statue of {layer.name} before : {layer.trainable}")
        layer.trainable = False
        print(f"trainable statue of {layer.name} after  : {layer.trainable}")


    base_layer = base_model.layers[:-1]
    # define the model and compile it
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="output_layer")
    )

    ## model.summary()
    logging.info(f"{STAGE} model summary: \n{_log_model_summary(new_model)}")

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3) # we can also just write "SGD" if we are fine with default learning rate
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)


    ## train the model 
    history = new_model.fit(X_train,
                        y_train_bin,
                        epochs=10,
                        validation_data=(X_valid, y_valid_bin),
                        verbose=2
                        )
    
    ## save the model
    model_dir_path = os.path.join("artifacts", "models")
    model_file_path = os.path.join(model_dir_path,"greater_than_5_model.h5")
    new_model.save(model_file_path)

    logging.info(f"new model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test, y_test_bin)}")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
        for i in range(5):
            logging.info(f"\n")
    except Exception as e:
        logging.exception(e)
        raise e