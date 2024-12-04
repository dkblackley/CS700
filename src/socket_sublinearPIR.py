### pir_socket_server.py ###
import socket
import pickle
import time
import json
from typing import List, Tuple, Set
import numpy as np
from dataclasses import dataclass
import logging
from sublinearPIR import PIRServer, PIRClient, ServerQuery
import secrets
import hashlib

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

class SocketPIRServer:
    def __init__(self, host: str = 'localhost', port: int = 12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.stats = NetworkStats()
        
    def initialize_database(self, size: int) -> List[int]:
        """Initialize random database of given size"""
        return [np.random.randint(0, 2) for _ in range(size)]

    def start(self):
        """Start the PIR server"""
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logging.info(f"Server listening on {self.host}:{self.port}")
        
        while True:
            conn, addr = self.socket.accept()
            logging.info(f"Connected by {addr}")
            self.handle_client(conn)

    def handle_client(self, conn: socket.socket):
        """Handle client connection"""
        try:
            # Reset stats for new connection
            self.stats = NetworkStats()
            self.stats.time_started = time.time()

            # Receive database size request
            size = self._receive_data(conn)
            database = self.initialize_database(size)
            pir_server = PIRServer(database)
            
            # Send acknowledgment
            self._send_data(conn, "ready")

            while True:
                # Receive query
                data = self._receive_data(conn)
                if not data or data == "exit":
                    break

                # Process query
                query = ServerQuery(data['punctured_set'])
                result = pir_server.answer_query(query)
                
                # Send result
                self._send_data(conn, result)

        except Exception as e:
            logging.error(f"Error handling client: {e}")
        finally:
            self.stats.time_ended = time.time()
            conn.close()
            self._log_stats()

    def _send_data(self, conn: socket.socket, data) -> None:
        """Send data with size tracking"""
        serialized = pickle.dumps(data)
        size = len(serialized)
        conn.send(len(serialized).to_bytes(8, 'big'))
        conn.send(serialized)
        self.stats.bytes_sent += size + 8

    def _receive_data(self, conn: socket.socket):
        """Receive data with size tracking"""
        size_bytes = conn.recv(8)
        if not size_bytes:
            return None
        size = int.from_bytes(size_bytes, 'big')
        data = b''
        while len(data) < size:
            chunk = conn.recv(min(size - len(data), 4096))
            if not chunk:
                return None
            data += chunk
        self.stats.bytes_received += len(data) + 8
        return pickle.loads(data)

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

### pir_socket_client.py ###
import socket
import pickle
import time
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)

class SocketPIRClient:
    def __init__(self, host: str = 'localhost', port: int = 12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stats = NetworkStats()
        
    def connect(self):
        """Connect to PIR server"""
        self.socket.connect((self.host, self.port))
        self.stats.time_started = time.time()

    def close(self):
        """Close connection and log stats"""
        self._send_data("exit")
        self.stats.time_ended = time.time()
        self.socket.close()
        self._log_stats()

    def initialize_pir(self, n: int) -> None:
        """Initialize PIR with database size n"""
        self._send_data(n)
        response = self._receive_data()
        if response != "ready":
            raise RuntimeError("Failed to initialize PIR")
        self.pir_client = PIRClient(n)

    def retrieve_bit(self, index: int, max_attempts: int = 10) -> Optional[int]:
        """Retrieve bit at given index"""
        # Execute offline phase
        hints = []
        for set_key in self.pir_client.sets:
            query = ServerQuery(set_key)
            self._send_data({"punctured_set": set_key})
            hint = self._receive_data()
            hints.append(hint)

        # Execute online phase
        attempts = 0
        while attempts < max_attempts:
            # Create query
            result = None
            for j, set_key in enumerate(self.pir_client.sets):
                elements = set_key.eval()
                if index not in elements:
                    continue

                b = secrets.randbelow(self.pir_client.n) < (self.pir_client.sqrt_n - 1)
                if b == 0:
                    punced_set = set_key.punc(index)
                    self._send_data({"punctured_set": punced_set})
                    answer = self._receive_data()
                    result = hints[j] ^ answer
                    return result
                else:
                    elements_list = list(elements - {index})
                    if not elements_list:
                        continue
                    rand_elem = elements_list[secrets.randbelow(len(elements_list))]
                    punced_set = set_key.punc(rand_elem)
                    self._send_data({"punctured_set": punced_set})
                    self._receive_data()  # Discard result
                    
            attempts += 1
            
        return None

    def _send_data(self, data) -> None:
        """Send data with size tracking"""
        serialized = pickle.dumps(data)
        size = len(serialized)
        self.socket.send(len(serialized).to_bytes(8, 'big'))
        self.socket.send(serialized)
        self.stats.bytes_sent += size + 8

    def _receive_data(self):
        """Receive data with size tracking"""
        size_bytes = self.socket.recv(8)
        size = int.from_bytes(size_bytes, 'big')
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(min(size - len(data), 4096))
            if not chunk:
                return None
            data += chunk
        self.stats.bytes_received += len(data) + 8
        return pickle.loads(data)

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

### test_pir_socket.py ###
def test_socket_pir():
    # Start server in a separate process
    from multiprocessing import Process
    server = SocketPIRServer()
    server_process = Process(target=server.start)
    server_process.start()
    time.sleep(1)  # Wait for server to start

    try:
        # Initialize client
        client = SocketPIRClient()
        client.connect()

        # Initialize PIR with database size
        n = 100
        client.initialize_pir(n)

        # Test retrieving several indices
        for test_i in range(5):
            logging.info(f"\nTrying to retrieve index {test_i}")
            result = client.retrieve_bit(test_i)
            if result is not None:
                logging.info(f"Retrieved bit: {result}")
            else:
                logging.info(f"Failed to retrieve bit at index {test_i}")

    finally:
        # Cleanup
        client.close()
        server_process.terminate()
        server_process.join()

if __name__ == "__main__":
    test_socket_pir()
