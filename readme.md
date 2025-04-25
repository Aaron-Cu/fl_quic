
# QUIC-Based Federated Learning Transport (Flower + aioquic)

This project demonstrates a federated learning system that replaces traditional gRPC-over-TCP communication with QUIC-over-UDP transport using aioquic, improving communication efficiency and scalability for federated learning.

## Requirements

- Python 3.10 or 3.11
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
```plaintext
your-project/
├── fl_quic_transport_flower.py
├── test_fl_quic_basic.py
├── requirements.txt
└── README.md
```
## Running

- **Start the QUIC Server**:
  ```bash
  python fl_quic_transport_flower.py server
  ```

- **Start a Client**:
  ```bash
  python fl_quic_transport_flower.py client
  ```

## Project Structure

| File | Description |
|:-----|:------------|
| `fl_quic_transport_flower.py` | Main server and client implementation using QUIC and Flower protobufs |
| `requirements.txt` | Project dependencies |
| `README.md` | Project overview and instructions |

## Citation

> A. Langley et al., "The QUIC Transport Protocol: Design and Internet-Scale Deployment," SIGCOMM 2017.


---

