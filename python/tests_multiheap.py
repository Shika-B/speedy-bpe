# Thanks chatgpt

import random
import time
from fast import MultisetHeap


def complex_stress_test():
    print("Starting complex stress test...")

    mh = MultisetHeap()
    counts = {}  # reference counts for verification
    keys = [f"key{i}" for i in range(1000)]  # pool of keys

    N = 50000  # total operations
    max_count = 100

    start_time = time.time()

    for i in range(N):
        op_type = random.choices(
            ["add", "sub", "popmax", "delete"], weights=[50, 30, 15, 5], k=1
        )[0]

        if op_type == "add":
            key = random.choice(keys)
            count = random.randint(1, max_count)
            mh.add(key, count)
            counts[key] = counts.get(key, 0) + count

        elif op_type == "sub" and counts:
            key = random.choice(list(counts.keys()))
            count = random.randint(1, counts[key])
            mh.sub(key, count)
            counts[key] -= count
            if counts[key] == 0:
                del counts[key]

        elif op_type == "popmax" and mh.heap:
            cnt, key = mh.popmax()
            # verify against reference
            assert key in counts and counts[key] == cnt, f"popmax mismatch: {key}"
            del counts[key]

        elif op_type == "delete" and mh.heap:
            key = random.choice(list(mh.d.keys()))
            mh.delete(key)
            if key in counts:
                del counts[key]

        # Occasionally perform deep validation
        if i % 5000 == 0 and mh.heap:
            # Check heap property
            for idx, (cnt, _) in enumerate(mh.heap):
                left = 2 * idx + 1
                right = 2 * idx + 2
                if left < len(mh.heap):
                    assert (
                        mh.heap[left][0] <= cnt
                    ), "Heap property violated (left child)"
                if right < len(mh.heap):
                    assert (
                        mh.heap[right][0] <= cnt
                    ), "Heap property violated (right child)"
            # Check dictionary mapping
            for k, pos in mh.d.items():
                assert mh.heap[pos][1] == k, f"Dictionary mapping mismatch for key {k}"

    end_time = time.time()
    print(f"Complex stress test passed in {end_time - start_time:.2f} seconds")
    print(f"Final heap size: {len(mh.heap)}, Remaining keys: {len(counts)}")


if __name__ == "__main__":
    complex_stress_test()
