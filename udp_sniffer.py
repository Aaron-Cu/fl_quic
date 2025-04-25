import socket

def udp_sniffer(host="0.0.0.0", port=8443):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print(f"[Sniffer] Listening for UDP packets on {host}:{port}...")

    while True:
        data, addr = sock.recvfrom(65535)
        print(f"[Sniffer] Received {len(data)} bytes from {addr}")

if __name__ == "__main__":
    udp_sniffer()
