import socket
import pickle
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import logging
from simplePIR import PIRParams, SimplePIR

logging.basicConfig(level=logging.INFO)


@dataclass
class NetworkStats:
    bytes_sent: int = 0
    bytes_received: int = 0
    time_started: float = 0
    time_ended: float = 0

    def get_data_rate(self) -> Tuple[float, float]:
        """Returns (upload_rate, download_rate) in bytes/second"""
        duration = self.time_ended - self.time_started
        if duration == 0:
            return 0.0, 0.0
        return self.bytes_sent / duration, self.bytes_received / duration


class PIRServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.db = None
        self.pir = None
        self.stats = NetworkStats()

    def initialize_database(self, params: PIRParams, N: int):
        """Initialize database with random values"""
        sqrt_N = int(np.sqrt(N))
        np.random.seed(0)
        self.db = np.random.randint(
            0, params.p, (sqrt_N, sqrt_N), dtype=np.int64)
        self.pir = SimplePIR(params)

    def start(self):
        """Start the PIR server"""
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logging.info(f"Server listening on {self.host}:{self.port}")

        while True:
            conn, addr = self.socket.accept()
            logging.info(f"Connected by {addr}")
            self.stats = NetworkStats()
            self.stats.time_started = time.time()

            try:
                while True:
                    size, cmd = self._receive_data(conn)
                    if cmd == "setup":
                        self.handle_setup(conn)
                    elif cmd == "answer":
                        self.handle_answer(conn)
                    elif cmd == "quit":
                        break
            except ConnectionResetError:
                logging.info("Client disconnected")
            finally:
                self.stats.time_ended = time.time()
                conn.close()
                self._log_stats()

    def handle_setup(self, conn):
        """Handle setup phase"""
        _, client_hint = self.pir.setup(self.db)

        setup_data = {
            'A': self.pir.A,
            'client_hint': client_hint
        }

        self._send_data(conn, setup_data)

    def handle_answer(self, conn):
        """Handle answer phase"""
        _, query = self._receive_data(conn)
        answer = self.pir.answer(self.db, None, query)
        self._send_data(conn, answer)

    def _send_data(self, conn, data) -> None:
        """Send data with size tracking"""
        serialized = pickle.dumps(data, protocol=4)
        size = len(serialized)
        conn.sendall(len(serialized).to_bytes(8, 'big'))
        conn.sendall(serialized)
        self.stats.bytes_sent += size + 8

    def _receive_data(self, conn):
        """Receive data with size tracking"""
        size_bytes = conn.recv(8)
        size = int.from_bytes(size_bytes, 'big')
        data = bytearray()
        while len(data) < size:
            chunk = conn.recv(min(size - len(data), 4096))
            if not chunk:
                raise ConnectionResetError()
            data.extend(chunk)
        self.stats.bytes_received += len(data) + 8
        return size, pickle.loads(bytes(data))

    def _log_stats(self):
        """Log network statistics"""
        upload_rate, download_rate = self.stats.get_data_rate()
        duration = self.stats.time_ended - self.stats.time_started
        logging.info(f"""
Connection Statistics:
Duration: {duration:.2f} seconds
Total Bytes Sent: {self.stats.bytes_sent}
Total Bytes Received: {self.stats.bytes_received}
Upload Rate: {upload_rate:.2f} bytes/second
Download Rate: {download_rate:.2f} bytes/second
""")


class PIRClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pir = None
        self.client_hint = None
        self.stats = NetworkStats()

    def connect(self):
        """Connect to PIR server"""
        self.socket.connect((self.host, self.port))
        self.stats.time_started = time.time()

    def initialize(self, params: PIRParams):
        """Initialize PIR client"""
        self.pir = SimplePIR(params)

    def query_index(self, i: int, db_dims: Tuple[int, int]) -> int:
        """Query database for index i"""
        # Setup phase
        self._send_data(self.socket, "setup")
        _, setup_data = self._receive_data(self.socket)

        self.pir.A = setup_data['A']
        self.client_hint = setup_data['client_hint']

        # Query phase
        state, query = self.pir.query(i, db_dims)
        self._send_data(self.socket, "answer")
        self._send_data(self.socket, query)
        _, answer = self._receive_data(self.socket)

        result = self.pir.recover(
            state, self.client_hint, answer, self.pir.params)

        return result

    def close(self):
        """Close connection to server"""
        self._send_data(self.socket, "quit")
        self.stats.time_ended = time.time()
        self.socket.close()
        self._log_stats()

    def _send_data(self, sock, data) -> None:
        """Send data with size tracking"""
        serialized = pickle.dumps(data, protocol=4)
        size = len(serialized)
        sock.sendall(len(serialized).to_bytes(8, 'big'))
        sock.sendall(serialized)
        self.stats.bytes_sent += size + 8

    def _receive_data(self, sock):
        """Receive data with size tracking"""
        size = int.from_bytes(sock.recv(8), 'big')
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(min(size - len(data), 4096))
            if not packet:
                raise ConnectionResetError()
            data.extend(packet)
        self.stats.bytes_received += len(data) + 8
        return size, pickle.loads(bytes(data))

    def _log_stats(self):
        """Log network statistics"""
        upload_rate, download_rate = self.stats.get_data_rate()
        duration = self.stats.time_ended - self.stats.time_started
        logging.info(f"""
Connection Statistics:
Duration: {duration:.2f} seconds
Total Bytes Sent: {self.stats.bytes_sent}
Total Bytes Received: {self.stats.bytes_received}
Upload Rate: {upload_rate:.2f} bytes/second
Download Rate: {download_rate:.2f} bytes/second
""")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # Initialize parameters
    params = PIRParams(
        n=1024,  # LWE dimension
        q=2**32,  # Ciphertext modulus
        p=991,   # Plaintext modulus
        sigma=6.4  # Gaussian parameter
    )

    # Start server in a separate thread
    server = PIRServer()
    N = 320 * 320
    server.initialize_database(params, N)

    import threading
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()

    # Give server time to start
    time.sleep(1)

    # Run client
    client = PIRClient()
    client.connect()
    client.initialize(params)

    # Test indices
    sqrt_N = int(np.sqrt(N))
    for test_idx in [42, 100, 1000]:
        result = client.query_index(test_idx, (sqrt_N, sqrt_N))
        i_row, i_col = test_idx // sqrt_N, test_idx % sqrt_N
        actual = server.db[i_row, i_col]
        logging.info(f"\nResults for index {test_idx}:")
        logging.info(f"Retrieved value: {result}")
        logging.info(f"Actual value: {actual}")
        logging.info(f"Correct: {result == actual}")
        logging.info("-" * 50)

    client.close()
