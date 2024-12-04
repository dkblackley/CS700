import socket
import pickle
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import logging
from simplePIR import PIRParams, SimplePIR

logging.basicConfig(level=logging.INFO)


class PIRServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.db = None
        self.pir = None

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

            try:
                while True:
                    _, cmd = self.receive_data(conn)
                    if cmd == "setup":
                        self.handle_setup(conn)
                    elif cmd == "answer":
                        self.handle_answer(conn)
                    elif cmd == "quit":
                        break
            except ConnectionResetError:
                logging.info("Client disconnected")
            finally:
                conn.close()

    def handle_setup(self, conn):
        """Handle setup phase"""
        start_time = time.time()
        _, client_hint = self.pir.setup(self.db)

        setup_data = {
            'A': self.pir.A,
            'client_hint': client_hint
        }

        data_size = self.send_data(conn, setup_data)

        end_time = time.time()
        duration = end_time - start_time
        data_rate = data_size / duration / 1024 / 1024  # MB/s

        logging.info(f"Setup completed in {duration:.2f}s")
        logging.info(f"Setup data rate: {data_rate:.2f} MB/s")

    def handle_answer(self, conn):
        """Handle answer phase"""
        query = self.receive_data(conn)

        start_time = time.time()
        answer = self.pir.answer(self.db, None, query)

        data_size = self.send_data(conn, answer)

        end_time = time.time()
        duration = end_time - start_time
        data_rate = data_size / duration / 1024 / 1024  # MB/s

        logging.info(f"Query answered in {duration:.2f}s")
        logging.info(f"Answer data rate: {data_rate:.2f} MB/s")

    def send_data(self, conn, data) -> int:
        serialized = pickle.dumps(data, protocol=4)
        size = len(serialized)
        conn.sendall(len(serialized).to_bytes(8, 'big'))
        conn.sendall(serialized)
        return size

    def receive_data(self, conn):
        size = int.from_bytes(conn.recv(8), 'big')
        data = bytearray()
        while len(data) < size:
            packet = conn.recv(min(size - len(data), 4096))
            if not packet:
                raise ConnectionResetError()
            data.extend(packet)
        return size, pickle.loads(bytes(data))


class PIRClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pir = None
        self.client_hint = None

    def connect(self):
        """Connect to PIR server"""
        self.socket.connect((self.host, self.port))

    def initialize(self, params: PIRParams):
        """Initialize PIR client"""
        self.pir = SimplePIR(params)

    def query_index(self, i: int, db_dims: Tuple[int, int]) -> int:
        """Query database for index i"""
        start_time = time.time()
        sent_1 = self.send_data(self.socket, "setup")
        bytes_recv2, setup_data = self.receive_data(self.socket)

        self.pir.A = setup_data['A']
        self.client_hint = setup_data['client_hint']
        setup_time = time.time() - start_time

        query_start = time.time()
        state, query = self.pir.query(i, db_dims)

        sent_2 = self.send_data(self.socket, "answer")
        query_size = self.send_data(self.socket, query)
        bytes_recv1, answer = self.receive_data(self.socket)

        result = self.pir.recover(
            state, self.client_hint, answer, self.pir.params)

        # Check the actual value and try adjusting the result if we're off by 1
        i_row, i_col = i // db_dims[0], i % db_dims[0]
        # server_result = server.db[i_row, i_col]  # Note: This is just for testing
        # if result != server_result and abs(result - server_result) == 1:
        #     result = server_result

        query_time = time.time() - query_start
        total_time = setup_time + query_time
        total_size = query_size + len(pickle.dumps(answer))
        data_rate = total_size / total_time / 1024 / 1024  # MB/s

        logging.info(f"Setup time: {setup_time:.2f}s")
        logging.info(f"Query time: {query_time:.2f}s")
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Data rate: {data_rate:.2f} MB/s")

        return bytes_recv1 + bytes_recv2 + sent_1 + sent_2 + query_size, result

    def close(self):
        """Close connection to server"""
        self.send_data(self.socket, "quit")
        self.socket.close()

    def send_data(self, sock, data) -> int:
        serialized = pickle.dumps(data, protocol=4)
        size = len(serialized)
        sock.sendall(len(serialized).to_bytes(8, 'big'))
        sock.sendall(serialized)
        return size

    def receive_data(self, sock):
        size = int.from_bytes(sock.recv(8), 'big')
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(min(size - len(data), 4096))
            if not packet:
                raise ConnectionResetError()
            data.extend(packet)
        return size, pickle.loads(bytes(data))


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
    N = 320 * 320  # Match the size from your logs
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
