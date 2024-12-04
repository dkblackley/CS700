import time
import numpy as np
from scipy import stats
from multiprocessing import Process, Manager
import logging
import json
from socket_simplePIR import PIRClient, PIRServer, PIRParams
from socket_sublinearPIR import SocketPIRServer as SublinearServer
from socket_sublinearPIR import SocketPIRClient as SublinearClient
import time

logging.basicConfig(level=logging.WARNING)


def calculate_required_samples(data, target_precision=0.05, confidence=0.95):
    s = np.std(data, ddof=1)
    mean = np.mean(data)
    if mean == 0 or s == 0:
        return 30
    t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    n = int(np.ceil((t_value * s / (mean * target_precision)) ** 2))
    return max(n, 30)


def run_server(server, stop_event):
    while not stop_event.is_set():
        try:
            server.start()
            time.sleep(0.1)  # time for server to spin up
        except BaseException:
            break


def benchmark_pir(N, pir_type='simple'):
    results = []
    initial_samples = 5

    # Phase 1: Initial samples
    while True:
        print(
            f"Running {pir_type} PIR sample {len(results)+1}/{initial_samples}")

        if pir_type == 'simple':
            params = PIRParams(n=1024, q=2**32, p=991, sigma=6.4)
            server = PIRServer()
            server.initialize_database(params, N)
        else:
            server = SublinearServer()
            server.initialize_database(N)  # Now passing size directly

        manager = Manager()
        stop_event = manager.Event()
        server_process = Process(target=run_server, args=(server, stop_event))
        server_process.daemon = True
        server_process.start()
        time.sleep(0.1)

        try:
            if pir_type == 'simple':
                client = PIRClient()
                client.connect()
                client.initialize(params)
            else:
                client = SublinearClient()
                client.connect()
                client.initialize_pir(N)

            index = np.random.randint(1, N - 1)

            start_time = time.time()
            if pir_type == 'simple':
                result = client.query_index(
                    index, (int(np.sqrt(N)), int(np.sqrt(N))))
            else:
                result = client.retrieve_bit(index)
            query_time = time.time() - start_time

            stats = client.stats
            results.append({
                'time': query_time,
                'bytes_sent': stats.bytes_sent,
                'bytes_received': stats.bytes_received
            })

        finally:
            stop_event.set()
            client.close()
            server_process.terminate()
            server_process.join(timeout=1)

        if len(results) >= initial_samples:
            times = [r['time'] for r in results]
            initial_samples = calculate_required_samples(times)
            if len(results) < initial_samples:
                break
    time.sleep(0.1)  # time for server to spin down
    return results


def calculate_stats(results):
    times = [r['time'] for r in results]
    bytes_sent = [r['bytes_sent'] for r in results]
    bytes_received = [r['bytes_received'] for r in results]

    conf_interval = 0.95
    t_value = stats.t.ppf((1 + conf_interval) / 2, len(times) - 1)

    def get_ci(data):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        margin = t_value * std / np.sqrt(len(data))
        return mean, mean - margin, mean + margin

    return {
        'time': {
            'mean': get_ci(times)[0],
            'ci_low': get_ci(times)[1],
            'ci_high': get_ci(times)[2]},
        'bytes_sent': np.mean(bytes_sent),
        'bytes_received': np.mean(bytes_received),
        'sample_count': len(results)}


def main():
    N_values = [4, 16]
    results = {}

    for N in N_values:
        print(f"\nTesting database size N={N}")
        results[N] = {
            'simple': calculate_stats(benchmark_pir(N, 'simple')),
            'sublinear': calculate_stats(benchmark_pir(N, 'sublinear'))
        }

        with open(f'pir_benchmark_N{N}.json', 'w') as f:
            json.dump(results[N], f, indent=2)


if __name__ == "__main__":
    main()
