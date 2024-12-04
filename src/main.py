import time
import numpy as np
import json
import cProfile
import pstats
from multiprocessing import Process, Manager
import logging
from scipy import stats
from dataclasses import asdict, dataclass
from socket_simplePIR import PIRClient, PIRServer
from socket_sublinearPIR import SocketPIRServer as SublinearSocketPIRServer
from socket_sublinearPIR import SocketPIRClient as SublinearSocketPIRClient
from simplePIR import PIRParams

logging.basicConfig(level=logging.WARNING)


@dataclass
class NetworkStats:
    bytes_sent: int = 0
    bytes_received: int = 0
    time_started: float = 0
    time_ended: float = 0


def calculate_required_samples(data, target_precision=0.05, confidence=0.95):
    """Calculate required samples for desired precision"""
    s = np.std(data, ddof=1)
    mean = np.mean(data)
    if mean == 0 or s == 0:
        return 30  # Return initial sample size if no variance

    t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    n = int(np.ceil((t_value * s / (mean * target_precision)) ** 2))

    # If we need fewer samples than we already have, stick with what we have
    return max(n, 30)


def run_server(server, stop_event):
    """Run server until stop event is set"""
    while not stop_event.is_set():
        try:
            server.start()
        except BaseException:
            break


def benchmark_simple_pir(N):
    """Benchmark SimplePIR implementation with Monte Carlo sampling"""
    results = []

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

    # First, collect 30 samples
    print(f"DEBUG: Initializing SimplePIR database of size {N} elements")
    print(f"DEBUG: A matrix will be size {int(np.sqrt(N))}x1024 elements")
    print(
        f"DEBUG: Total memory for A matrix (approximate): {(int(np.sqrt(N))*1024*8)/(1024*1024):.2f} MB")

    # Phase 1: Initial 30 samples
    print("\nPhase 1: Collecting initial 30 samples...")
    for i in range(30):
        print(f"Running initial SimplePIR sample {i+1}/30")

        params = PIRParams(n=1024, q=2**32, p=991, sigma=6.4)
        server = PIRServer()
        server.initialize_database(params, N)

        manager = Manager()
        stop_event = manager.Event()

        server_process = Process(target=run_server, args=(server, stop_event))
        server_process.daemon = True
        server_process.start()
        time.sleep(1)

        try:
            client = PIRClient()
            client.connect()
            client.initialize(params)

            index = np.random.randint(0, N)

            start_time = time.time()
            result = client.query_index(
                index, (int(np.sqrt(N)), int(np.sqrt(N))))
            query_time = time.time() - start_time

            setup_bytes = 8 + int(np.sqrt(N)) * 1024 * 8
            query_bytes = int(np.sqrt(N)) * 8
            answer_bytes = int(np.sqrt(N)) * 8

            results.append({
                'time': query_time,
                'bytes_sent': setup_bytes + query_bytes,
                'bytes_received': answer_bytes
            })

        finally:
            stop_event.set()
            try:
                client.close()
            except BaseException:
                pass
            try:
                server_process.terminate()
                server_process.join(timeout=1)
            except BaseException:
                pass

    # Phase 2: Calculate required samples and collect more if needed
    times = [r['time'] for r in results]
    required_samples = calculate_required_samples(times)

    if required_samples > 30:
        print(
            f"\nPhase 2: Need {required_samples-30} more samples for 95% confidence")
        current_samples = 30

        while current_samples < required_samples:
            print(
                f"Running additional SimplePIR sample {current_samples+1}/{required_samples}")

            params = PIRParams(n=1024, q=2**32, p=991, sigma=6.4)
            server = PIRServer()
            server.initialize_database(params, N)

            manager = Manager()
            stop_event = manager.Event()

            server_process = Process(
                target=run_server, args=(
                    server, stop_event))
            server_process.daemon = True
            server_process.start()
            time.sleep(1)

            try:
                client = PIRClient()
                client.connect()
                client.initialize(params)

                index = np.random.randint(0, N)

                start_time = time.time()
                result = client.query_index(
                    index, (int(np.sqrt(N)), int(np.sqrt(N))))
                query_time = time.time() - start_time

                setup_bytes = 8 + int(np.sqrt(N)) * 1024 * 8
                query_bytes = int(np.sqrt(N)) * 8
                answer_bytes = int(np.sqrt(N)) * 8

                results.append({
                    'time': query_time,
                    'bytes_sent': setup_bytes + query_bytes,
                    'bytes_received': answer_bytes
                })

                current_samples += 1

            finally:
                stop_event.set()
                try:
                    client.close()
                except BaseException:
                    pass
                try:
                    server_process.terminate()
                    server_process.join(timeout=1)
                except BaseException:
                    pass

    print(f"\nFinal sample count: {len(results)}")
    return results


def benchmark_sublinear_pir(N):
    """Benchmark SublinearPIR implementation with Monte Carlo sampling"""
    results = []

    print(f"Using database size {N} for SublinearPIR")

    # Phase 1: Initial 30 samples
    print("\nPhase 1: Collecting initial 30 samples...")
    while len(results) < 30:
        print(f"Running initial SublinearPIR sample {len(results)+1}/30")
        try:
            # Initialize server with a random database
            server = SublinearSocketPIRServer()
            # Create a random database of size N with binary values
            database = [np.random.randint(0, 2) for _ in range(N)]
            # Make sure this method exists
            server.initialize_database(database)

            manager = Manager()
            stop_event = manager.Event()

            server_process = Process(
                target=run_server, args=(
                    server, stop_event))
            server_process.daemon = True
            server_process.start()
            time.sleep(1)  # Give server time to start

            client = SublinearSocketPIRClient()
            client.connect()
            client.initialize_pir(N)

            # Test server connection
            if not client.is_connected():
                raise ConnectionError("Failed to connect to server")

            index = np.random.randint(0, N)
            print(f"Querying index {index}")

            start_time = time.time()
            result = client.retrieve_bit(index)
            query_time = time.time() - start_time

            if result is None:
                print(f"Warning: Got None result for index {index}")
                continue

            stats = client.stats
            print(f"Query successful: got result {result}")
            results.append({
                'time': query_time,
                'bytes_sent': stats.bytes_sent,
                'bytes_received': stats.bytes_received,
                'index': index,
                'result': result
            })

        except Exception as e:
            print(f"Error in SublinearPIR sample: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            stop_event.set()
            try:
                client.close()
            except BaseException:
                pass
            try:
                server_process.terminate()
                server_process.join(timeout=1)
            except BaseException:
                pass

    # Phase 2: Calculate required samples and collect more if needed
    times = [r['time'] for r in results]
    required_samples = calculate_required_samples(times)

    if required_samples > 30:
        print(
            f"\nPhase 2: Need {required_samples-30} more samples for 95% confidence")
    while len(results) < required_samples:
        print(
            f"Running additional SublinearPIR sample {len(results)+1}/{required_samples}")
        try:
            server = SublinearSocketPIRServer()
            manager = Manager()
            stop_event = manager.Event()

            server_process = Process(
                target=run_server, args=(
                    server, stop_event))
            server_process.daemon = True
            server_process.start()
            time.sleep(1)

            client = SublinearSocketPIRClient()
            client.connect()
            client.initialize_pir(N)

            index = np.random.randint(0, N)

            start_time = time.time()
            result = client.retrieve_bit(index)
            query_time = time.time() - start_time

            if result is not None:
                stats = client.stats
                results.append({
                    'time': query_time,
                    'bytes_sent': stats.bytes_sent,
                    'bytes_received': stats.bytes_received
                })

        except Exception as e:
            print(f"Error in SublinearPIR sample: {str(e)}")
            continue

        finally:
            stop_event.set()
            try:
                client.close()
            except BaseException:
                pass
            try:
                server_process.terminate()
                server_process.join(timeout=1)
            except BaseException:
                pass

    print(f"\nFinal sample count: {len(results)}")
    return results


def main():
    N_values = [4, 16, 64]
    results = {
        'simple_pir': {},
        'sublinear_pir': {}
    }

    profiler = cProfile.Profile()

    for N in N_values:
        print(f"\nTesting database size N={N}")

        # SimplePIR
        print("Running SimplePIR benchmarks...")
        profiler.enable()
        simple_results = benchmark_simple_pir(N)
        profiler.disable()
        results['simple_pir'][N] = simple_results
        print(
            f"\nSimplePIR complete - collected {len(simple_results)} samples")

        # SublinearPIR
        print("\nRunning SublinearPIR benchmarks...")
        profiler.enable()
        sublinear_results = benchmark_sublinear_pir(N)
        profiler.disable()
        results['sublinear_pir'][N] = sublinear_results
        print(
            f"\nSublinearPIR complete - collected {len(sublinear_results)} samples")

        # Save intermediate results
        with open(f'pir_benchmark_results_N{N}.json', 'w') as f:
            json.dump(results, f)

    # Save profiling results
    with open('profile_stats.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()


if __name__ == "__main__":
    main()
