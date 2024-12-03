import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


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
        print("\n=== Setup Phase ===")
        print(f"Database shape: {db.shape}")
        print(f"Database min/max values: {np.min(db)}/{np.max(db)}")

        sqrt_N = db.shape[0]
        # Sample random matrix A
        self.A = np.random.randint(0, self.params.q, (sqrt_N, self.params.n))
        print(f"A matrix shape: {self.A.shape}")
        print(f"A matrix min/max values: {np.min(self.A)}/{np.max(self.A)}")

        # Compute hint = DB · A
        client_hint = np.mod(db @ self.A, self.params.q)
        print(f"Client hint shape: {client_hint.shape}")
        print(
            f"Client hint min/max values: {np.min(client_hint)}/{np.max(client_hint)}")

        return None, client_hint

    def query(self,
              i: int,
              db_dims: Tuple[int,
                             int]) -> Tuple[Tuple[int,
                                                  np.ndarray],
                                            np.ndarray]:
        """Generate query for index i"""
        print("\n=== Query Phase ===")
        sqrt_N = db_dims[0]
        i_row, i_col = i // sqrt_N, i % sqrt_N
        print(f"Querying index {i} (row={i_row}, col={i_col})")

        # Sample secret and error
        s = np.random.randint(0, self.params.q, self.params.n)
        print(f"Secret s shape: {s.shape}")
        print(f"Secret s min/max values: {np.min(s)}/{np.max(s)}")

        e = self._sample_gaussian((sqrt_N,))
        print(f"Error e shape: {e.shape}")
        print(f"Error e min/max values: {np.min(e)}/{np.max(e)}")

        # Generate unit vector
        u = np.zeros(sqrt_N)
        u[i_col] = 1

        # Compute As + e + Δ·u
        if self.A is None:  # Use same A from setup
            self.A = np.random.randint(
                0, self.params.q, (sqrt_N, self.params.n))

        query = np.mod(self.A @ s + e + self.params.delta * u, self.params.q)
        print(f"Query vector shape: {query.shape}")
        print(f"Query vector min/max values: {np.min(query)}/{np.max(query)}")
        print(f"Delta value: {self.params.delta}")

        return (i_row, s), query

    def answer(
            self,
            db: np.ndarray,
            server_hint: None,
            query: np.ndarray) -> np.ndarray:
        """Server's answer computation"""
        print("\n=== Answer Phase ===")
        # Compute DB · query
        answer = np.mod(db @ query, self.params.q)
        print(f"Answer vector shape: {answer.shape}")
        print(
            f"Answer vector min/max values: {np.min(answer)}/{np.max(answer)}")
        return answer

    def recover(self, state: Tuple[int, np.ndarray], client_hint: np.ndarray,
                answer: np.ndarray, params: PIRParams) -> int:
        """Recover the requested database element"""
        print("\n=== Recovery Phase ===")
        i_row, s = state

        # Compute answer[i_row] - client_hint[i_row] · s
        hint_part = np.mod(client_hint[i_row] @ s, self.params.q)
        answer_part = answer[i_row]
        print(f"Answer part value: {answer_part}")
        print(f"Hint part value: {hint_part}")

        noised = answer_part - hint_part
        noised = noised % self.params.q
        print(f"Noised value: {noised}")

        # Round to nearest multiple of Δ
        delta = self.params.delta
        denoised = round(noised / delta) * delta
        print(f"Denoised value: {denoised}")
        result = (denoised // delta) % self.params.p
        print(f"Final result (mod p): {result}")

        return result


# Example usage:
if __name__ == "__main__":
    print("=== SimplePIR Parameters ===")
    params = PIRParams(
        n=1024,  # LWE dimension
        q=2**32,  # Ciphertext modulus
        p=991,   # Plaintext modulus
        sigma=6.4  # Gaussian parameter
    )
    print(f"n: {params.n}")
    print(f"q: {params.q}")
    print(f"p: {params.p}")
    print(f"sigma: {params.sigma}")
    print(f"delta: {params.delta}")

    pir = SimplePIR(params)

    # Create example database
    N = 1024  # Total DB size
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
    print("\n=== Final Results ===")
    print(f"Retrieved value: {result}")
    print(f"Actual value: {db[i_row, i_col]}")
    print(f"Correct: {result == db[i_row, i_col]}")
