import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import logging


@dataclass
class PIRParams:
    """Parameters for the SimplePIR scheme"""
    n: int  # LWE secret dimension
    q: int  # Ciphertext modulus
    p: int  # Plaintext modulus
    sigma: float  # Gaussian error parameter

    @property
    def delta(self) -> int:
        """Returns ⌊q/p⌋ as defined in the paper"""
        return self.q // self.p


class SimplePIR:
    def __init__(self, params: PIRParams):
        self.params = params
        self.A = None  # Store A matrix for debugging

    def _sample_gaussian(self, size: tuple) -> np.ndarray:
        """Sample from discrete Gaussian distribution"""
        return np.random.normal(0, self.params.sigma, size).astype(int)

    def setup(self, db: np.ndarray) -> Tuple[None, np.ndarray]:
        """Setup phase of SimplePIR"""
        logging.info("\n=== Setup Phase ===")
        logging.info(f"Database shape: {db.shape}")
        logging.info(f"Database min/max values: {np.min(db)}/{np.max(db)}")

        sqrt_N = db.shape[0]
        # Sample random matrix A
        self.A = np.random.randint(0, self.params.q, (sqrt_N, self.params.n))
        logging.info(f"A matrix shape: {self.A.shape}")
        logging.info(
            f"A matrix min/max values: {np.min(self.A)}/{np.max(self.A)}")

        # Compute hint = DB · A
        client_hint = np.mod(db @ self.A, self.params.q)
        logging.info(f"Client hint shape: {client_hint.shape}")
        logging.info(
            f"Client hint min/max values: {np.min(client_hint)}/{np.max(client_hint)}")

        return None, client_hint

    def query(self,
              i: int,
              db_dims: Tuple[int,
                             int]) -> Tuple[Tuple[int,
                                                  np.ndarray],
                                            np.ndarray]:
        """Generate query for index i"""
        logging.info("\n=== Query Phase ===")
        sqrt_N = db_dims[0]
        i_row, i_col = i // sqrt_N, i % sqrt_N
        logging.info(f"Querying index {i} (row={i_row}, col={i_col})")

        # Sample secret and error
        s = np.random.randint(0, self.params.q, self.params.n)
        logging.info(f"Secret s shape: {s.shape}")
        logging.info(f"Secret s min/max values: {np.min(s)}/{np.max(s)}")

        e = self._sample_gaussian((sqrt_N,))
        logging.info(f"Error e shape: {e.shape}")
        logging.info(f"Error e min/max values: {np.min(e)}/{np.max(e)}")

        # Generate unit vector
        u = np.zeros(sqrt_N)
        u[i_col] = 1

        # Compute As + e + Δ·u
        if self.A is None:  # Use same A from setup
            self.A = np.random.randint(
                0, self.params.q, (sqrt_N, self.params.n))

        query = np.mod(self.A @ s + e + self.params.delta * u, self.params.q)
        logging.info(f"Query vector shape: {query.shape}")
        logging.info(
            f"Query vector min/max values: {np.min(query)}/{np.max(query)}")
        logging.info(f"Delta value: {self.params.delta}")

        return (i_row, s), query

    def answer(
            self,
            db: np.ndarray,
            server_hint: None,
            query: np.ndarray) -> np.ndarray:
        """Server's answer computation"""
        logging.info("\n=== Answer Phase ===")
        # Compute DB · query
        answer = np.mod(db @ query, self.params.q)
        logging.info(f"Answer vector shape: {answer.shape}")
        logging.info(
            f"Answer vector min/max values: {np.min(answer)}/{np.max(answer)}")
        return answer

    def recover(self, state: Tuple[int, np.ndarray], client_hint: np.ndarray,
                answer: np.ndarray, params: PIRParams) -> int:
        """Recover the requested database element"""
        logging.info("\n=== Recovery Phase ===")
        i_row, s = state

        # Compute answer[i_row] - client_hint[i_row] · s
        hint_part = np.mod(client_hint[i_row] @ s, self.params.q)
        answer_part = answer[i_row]
        logging.info(f"Answer part value: {answer_part}")
        logging.info(f"Hint part value: {hint_part}")

        noised = answer_part - hint_part
        noised = noised % self.params.q
        logging.info(f"Noised value: {noised}")

        # Round to nearest multiple of Δ
        delta = self.params.delta
        denoised = round(noised / delta) * delta
        logging.info(f"Denoised value: {denoised}")
        result = (denoised // delta) % self.params.p
        logging.info(f"Final result (mod p): {result}")

        return result


# Example usage:
if __name__ == "__main__":
    logging.info("=== SimplePIR Parameters ===")
    params = PIRParams(
        n=1024,  # LWE dimension
        q=2**32,  # Ciphertext modulus
        p=991,   # Plaintext modulus
        sigma=6.4  # Gaussian parameter
    )
    logging.info(f"n: {params.n}")
    logging.info(f"q: {params.q}")
    logging.info(f"p: {params.p}")
    logging.info(f"sigma: {params.sigma}")
    logging.info(f"delta: {params.delta}")

    pir = SimplePIR(params)

    # Create example database
    N = 1024 * 100  # Total DB size
    sqrt_N = int(np.sqrt(N))
    db = np.random.randint(0, params.p, (sqrt_N, sqrt_N))

    # Run protocol
    i = 42  # Index to query

    # Setup
    server_hint, client_hint = pir.setup(db)

    # Query generation
    state, query = pir.query(i, db.shape)

    # Server answer
    answer = pir.answer(db, server_hint, query)

    # Recovery
    result = pir.recover(state, client_hint, answer, params)

    # Verify correctness
    i_row, i_col = i // sqrt_N, i % sqrt_N
    logging.info("\n=== Final Results ===")
    logging.info(f"Retrieved value: {result}")
    logging.info(f"Actual value: {db[i_row, i_col]}")
    logging.info(f"Correct: {result == db[i_row, i_col]}")
