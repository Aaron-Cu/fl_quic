import asyncio
import tensorflow as tf
from flwr_datasets import FederatedDataset
from fl_quic_transport_flower import FLQuicServer, fl_client_quic

async def run_test():
    # Start the server
    server = FLQuicServer()
    server_task = asyncio.create_task(server.run())
    print("[Test] Server started.")

    # Build a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 data
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id=0, split="train")
    partition.set_format("numpy")
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]

    # Run the client after a short delay to allow server to start
    await asyncio.sleep(1)
    await fl_client_quic("localhost", 4433, model, x_train, y_train)

# Entry point
if __name__ == "__main__":
    asyncio.run(run_test())
