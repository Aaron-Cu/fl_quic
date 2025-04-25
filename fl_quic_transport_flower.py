# File: fl_quic_transport_flower.py

import asyncio
import pickle
import time
from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration

from aioquic.asyncio.protocol import QuicConnectionProtocol

class FLQuicProtocol(QuicConnectionProtocol):
    def __init__(self, *args, server: "FLQuicServer", **kwargs):
        super().__init__(*args, **kwargs)
        self._server = server

    async def stream_handler(self, stream_id: int, reader, writer):
        await self._server.handle_stream(stream_id, reader)

    def quic_event_received(self, event):
        if isinstance(event, HandshakeCompleted):
            print("[Server] QUIC handshake completed.")
        elif isinstance(event, StreamDataReceived):
            # StreamDataReceived is automatically passed to stream_handler
            pass

# ------------------ Minimal Fake Proto Simulation ------------------

from dataclasses import dataclass, field, asdict
from typing import List, Dict

@dataclass
class Parameters:
    tensors: List[bytes] = field(default_factory=list)
    tensor_type: str = ""

@dataclass
class FitRes:
    num_examples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Parameters = field(default_factory=Parameters)

    def SerializeToString(self) -> bytes:
        return pickle.dumps(asdict(self))

    def ParseFromString(self, data: bytes):
        obj = pickle.loads(data)
        self.num_examples = obj["num_examples"]
        self.metrics = obj["metrics"]
        self.parameters = Parameters(**obj["parameters"])

@dataclass
class FitIns:
    parameters: Parameters = field(default_factory=Parameters)

    def SerializeToString(self) -> bytes:
        return pickle.dumps(asdict(self))

    def ParseFromString(self, data: bytes):
        obj = pickle.loads(data)
        self.parameters = Parameters(**obj["parameters"])

# Alias for code compatibility
class fed:
    FitRes = FitRes
    FitIns = FitIns

# ------------------------- Helper Functions -------------------------

def serialize_proto(proto_msg):
    return proto_msg.SerializeToString()

def deserialize_proto(data, msg_type):
    msg = msg_type()
    msg.ParseFromString(data)
    return msg

# ------------------------- FL CLIENT (QUIC) -------------------------

async def fl_client_quic(host, port, model, x_train, y_train):
    configuration = QuicConfiguration(is_client=True)
    configuration.verify_mode = None
    configuration.verify_certificate = False
    configuration.is_client = True
    configuration.create_default_certificates = False
    configuration.server_name = "localhost"
    configuration.alpn_protocols = ["hq-29"]

    with open("ca_cert.pem", "rb") as f:
        configuration.load_verify_locations(cadata=f.read().decode())

    async with connect(
        host, 
        port, 
        configuration=configuration, 
        session_ticket_handler=None
    ) as client:
        quic_stream = await client.create_stream()


        # Step 1: Train locally
        print("[Client] Starting local training...")
        start_train = time.time()

        batch_size = 32
        n_samples = x_train.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(1):  # One epoch
            for i in range(0, n_samples, batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                model.train_on_batch(x_batch, y_batch)

        end_train = time.time()
        print(f"[Client] Local training completed in {end_train - start_train:.2f} seconds.")

        weights = model.get_weights()

        # Step 2: Create FitRes
        fitres = fed.FitRes()
        fitres.num_examples = len(x_train)
        fitres.metrics["accuracy"] = float(model.evaluate(x_train, y_train, verbose=0)[1])
        fitres.parameters.tensors.extend([pickle.dumps(w) for w in weights])
        fitres.parameters.tensor_type = "weights_pickle"

        # Step 3: Send to server
        serialized = serialize_proto(fitres)
        print(f"[Client] Sending FitRes ({len(serialized)} bytes)...")
        quic_stream.write(serialized)
        await quic_stream._send_stream_data(end_stream=True)

        # Step 4: Receive FitIns
        response = await quic_stream.read()
        fitins = deserialize_proto(response, fed.FitIns)
        received_weights = [pickle.loads(t) for t in fitins.parameters.tensors]
        model.set_weights(received_weights)

        print(f"[Client] Received updated global model ({len(response)} bytes).")

# ------------------------- FL SERVER (QUIC) -------------------------

class FLQuicServer:
    def __init__(self):
        self.received_updates = []

    async def handle_stream(self, stream_id, quic_stream):
        print("[Server] New client connected.")
        data = await quic_stream.read()
        fitres = deserialize_proto(data, fed.FitRes)
        self.received_updates.append(fitres)

        print(f"[Server] Received FitRes: {fitres.num_examples} examples, "
              f"Accuracy {fitres.metrics['accuracy']:.4f}")

        received_weights = [pickle.loads(t) for t in fitres.parameters.tensors]
        fitins = fed.FitIns()
        fitins.parameters.tensors.extend([pickle.dumps(w) for w in received_weights])
        fitins.parameters.tensor_type = "weights_pickle"

        serialized = serialize_proto(fitins)
        await quic_stream.write(serialized)
        print(f"[Server] Sent FitIns ({len(serialized)} bytes) back to client.")

    async def run(self, host="0.0.0.0", port=4433):
        configuration = QuicConfiguration(is_client=False)
        configuration.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        configuration.verify_mode = None
        configuration.alpn_protocols = ["hq-29"]

        from aioquic.asyncio.protocol import QuicConnectionProtocol

        await serve(
            host,
            port,
            configuration=configuration,
            create_protocol=lambda *args, **kwargs: FLQuicProtocol(*args, server=self, **kwargs),
            stream_handler=None 
        )


        print(f"[Server] QUIC server running at {host}:{port}")

        # Hold forever manually
        while True:
            await asyncio.sleep(3600)



# ------------------------- ENTRY POINT -------------------------

if __name__ == '__main__':
    import sys
    import tensorflow as tf
    from flwr_datasets import FederatedDataset

    if len(sys.argv) < 2:
        print("Usage: python fl_quic_transport_flower.py [server|client]")
        exit(1)

    if sys.argv[1] == "server":
        server = FLQuicServer()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server.run())

    elif sys.argv[1] == "client":
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
        partition = fds.load_partition(partition_id=0, split="train")
        partition.set_format("numpy")
        partition = partition.train_test_split(test_size=0.2, seed=42)
        x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]

        asyncio.run(fl_client_quic("127.0.0.1", 4433, model, x_train, y_train))  # <-- Connect to 127.0.0.1!

    else:
        print("Invalid argument. Use 'server' or 'client'.")
