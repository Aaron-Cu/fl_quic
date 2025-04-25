import asyncio
import tensorflow as tf
from flwr_datasets import FederatedDataset
from fl_quic_transport_flower import fl_client_quic

async def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id=0, split="train")
    partition.set_format("numpy")
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]

    await fl_client_quic("localhost", 8443, model, x_train, y_train)

asyncio.run(main())