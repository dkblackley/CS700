import secrets
import hashlib
import numpy as np
from typing import List, Optional, Set

class PIRServer:
    def __init__(self, database: List[int]):
        self.database = np.array(database)
        
    def answer_query(self, query: 'ServerQuery') -> int:
        """Answer a PIR query by computing parity of accessed elements"""
        indices = query.punctured_set.eval()
        result = sum(self.database[i] for i in indices) % 2
        print(f"Server computing parity for indices {sorted(indices)}: {result}")
        print(f"Values at those indices: {[self.database[i] for i in sorted(indices)]}")
        return result

class PIRClient:
    def __init__(self, n: int):
        """Initialize client for database of size n"""
        self.n = n
        self.sqrt_n = int(np.sqrt(n))
        self.m = max(1, int((self.n / self.sqrt_n) * np.log(n)))
        
        print(f"\nInitializing client with:")
        print(f"n = {n}, sqrt_n = {self.sqrt_n}, m = {self.m}")
        
        # Initialize m random sets of size √n
        self.sets = []
        for i in range(self.m):
            set_key = PseudorandomSet(n, self.sqrt_n)
            print(f"Generated set {i} with {len(set_key.eval())} elements")
            self.sets.append(set_key)

    def offline_phase(self, server: PIRServer) -> List[int]:
        """Execute offline phase, getting hints from server"""
        print("\nStarting offline phase")
        hints = []
        for idx, set_key in enumerate(self.sets):
            query = ServerQuery(set_key)
            hint = server.answer_query(query)
            print(f"Set {idx} elements: {sorted(set_key.eval())}")
            print(f"Received hint: {hint}")
            hints.append(hint)
        return hints
    
    def online_phase(self, i: int, hints: List[int], server: PIRServer) -> Optional[int]:
        """Execute online phase to retrieve element i"""
        print(f"\nStarting online phase for index {i}")
        
        # Find a set containing i
        for j, set_key in enumerate(self.sets):
            elements = set_key.eval()
            print(f"\nChecking set {j}: {sorted(list(elements))}")
            
            if i not in elements:
                print(f"Index {i} not in set {j}")
                continue
                
            print(f"Found index {i} in set {j}")
            
            # Create random bit b that equals 0 with probability (s-1)/n
            b = secrets.randbelow(self.n) < (self.sqrt_n - 1)
            print(f"Generated random bit b = {b}")
            
            if b == 0:
                print("Puncturing at target index")
                punced_set = set_key.punc(i)
                query = ServerQuery(punced_set)
                answer = server.answer_query(query)
                print(f"Original hint: {hints[j]}")
                print(f"Server answer: {answer}")
                result = hints[j] ^ answer
                print(f"Computing result as {hints[j]} XOR {answer} = {result}")
                return result
            else:
                print("Puncturing at random index")
                elements_list = list(elements - {i})
                if not elements_list:
                    print("No other elements to choose from")
                    continue
                rand_elem = elements_list[secrets.randbelow(len(elements_list))]
                print(f"Chose random element: {rand_elem}")
                punced_set = set_key.punc(rand_elem)
                return None
                
        print(f"Failed to find index {i} in any set")
        return None

class PseudorandomSet:
    """A puncturable pseudorandom set implementation"""
    def __init__(self, size: int, set_size: int, initial_elements: Optional[Set[int]] = None):
        if set_size > size:
            raise ValueError("Set size cannot be larger than universe size")
        self.size = size
        self.set_size = set_size
        self.prf = PuncturablePRF(size)
        self.excluded_elements = set()  # Track excluded elements
        self.initial_elements = initial_elements
        
    def eval(self) -> Set[int]:
        """Evaluate the set"""
        if self.initial_elements is not None:
            return self.initial_elements - self.excluded_elements

        elements = set()
        attempt = 0
        while len(elements - self.excluded_elements) < self.set_size and attempt < self.size * 2:
            val = self.prf.eval(attempt)
            if val not in self.excluded_elements:
                elements.add(val)
            attempt += 1
            
        result = elements - self.excluded_elements
        # Ensure we have exactly set_size elements
        if len(result) > self.set_size:
            result = set(sorted(list(result))[:self.set_size])
        return result
        
    def punc(self, element: int) -> 'PseudorandomSet':
        """Create a punctured set that excludes given element"""
        if not 0 <= element < self.size:
            raise ValueError(f"Element {element} out of range [0, {self.size})")
            
        current_elements = self.eval()
        if element not in current_elements:
            raise ValueError(f"Cannot puncture element {element} that is not in the set")
            
        new_set = PseudorandomSet(self.size, self.set_size - 1, current_elements)
        new_set.excluded_elements.add(element)
        return new_set

class ServerQuery:
    def __init__(self, punctured_set: 'PseudorandomSet'):
        self.punctured_set = punctured_set


class PuncturablePRF:
    def __init__(self, size: int):
        self.size = size
        self.key = secrets.token_bytes(32)
        
    def eval(self, x: int) -> int:
        """Evaluate PRF at point x"""
        if not 0 <= x < self.size:
            raise ValueError(f"Input {x} out of range [0, {self.size})")
        
        h = hashlib.sha256(self.key + x.to_bytes(4, 'big')).digest()
        return int.from_bytes(h[:4], 'big') % self.size
        
    def punc(self, x: int) -> 'PuncturablePRF':
        """Create punctured key that can't evaluate at x"""
        if not 0 <= x < self.size:
            raise ValueError(f"Input {x} out of range [0, {self.size})")
            
        new_prf = PuncturablePRF(self.size)
        new_prf.key = hashlib.sha256(self.key + x.to_bytes(4, 'big')).digest()[:32]
        return new_prf

def test_simple_pir():
    # Create a small test database
    n = 16  # database size
    database = [secrets.randbelow(2) for _ in range(n)]
    print(f"Database: {database}")

    # Setup PIR server and client
    server = PIRServer(database)
    client = PIRClient(n)

    # Print initial sets for debugging
    print("\nInitial sets:")
    for idx, s in enumerate(client.sets):
        elements = s.eval()
        print(f"Set {idx}: {sorted(elements)}")

    # Execute offline phase
    hints = client.offline_phase(server)
    print(f"\nReceived {len(hints)} hints:")
    for idx, h in enumerate(hints):
        print(f"Hint {idx}: {h}")

    # Test retrieving several indices
    for test_i in range(min(5, n)):
        print(f"\n{'='*50}")
        print(f"Trying to retrieve index {test_i}")
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            result = client.online_phase(test_i, hints, server)
            if result is not None:
                print(f"Retrieved bit: {result}")
                print(f"Actual bit: {database[test_i]}")
                assert result == database[test_i], f"Mismatch at index {test_i}"
                break
            attempts += 1
            print("Retrying...")
        
        if attempts == max_attempts:
            print(f"Failed to retrieve index {test_i} after {max_attempts} attempts")

if __name__ == "__main__":
    test_simple_pir()
