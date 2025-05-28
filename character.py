import argparse
from functools import lru_cache

def parse_cycle_type(cycle_counts):
    """Convert cycle count vector to a list of cycle lengths."""
    cycle_lengths = []
    for length, count in enumerate(cycle_counts, start=1):
        cycle_lengths.extend([length] * count)
    return cycle_lengths

def character_M(partition, cycle_lengths):
    """
    Compute χ_{M^λ}(σ): number of λ-tabloids fixed by a permutation with given cycle lengths.
    Uses backtracking to assign each cycle to a row of the partition.
    """
    rows = len(partition)
    
    @lru_cache(None)
    def backtrack(idx, remaining):
        if idx == len(cycle_lengths):
            return 1 if all(r == 0 for r in remaining) else 0
        total = 0
        length = cycle_lengths[idx]
        for i in range(rows):
            if remaining[i] >= length:
                new_rem = list(remaining)
                new_rem[i] -= length
                total += backtrack(idx + 1, tuple(new_rem))
        return total
    
    return backtrack(0, tuple(partition))


def rim_hooks(shape, hook_length):
    """
    Enumerate all removable rim hooks of given length from the Young diagram 'shape'.
    'shape' is a tuple of row lengths. Returns a list of (new_shape, height) pairs.
    """
    hooks = []
    # Convert shape to a set of coordinates (row, col)
    cells = {(r, c) for r, length in enumerate(shape) for c in range(length)}
    
    # Identify outer corners: cells with no neighbor to right and no neighbor below
    corners = [(r, c) for (r, c) in cells
               if (r, c+1) not in cells and (r+1, c) not in cells]
    
    def is_border(cell):
        r, c = cell
        # A border cell has at least one missing neighbor to its right or below
        return (r, c+1) not in cells or (r+1, c) not in cells
    
    # DFS to find rim hooks
    def dfs(path, visited):
        if len(path) == hook_length:
            # Check if removal yields a valid partition
            new_cells = cells - set(path)
            # Reconstruct new shape
            new_shape = []
            for r in range(len(shape)):
                row_len = sum(1 for (rr, _) in new_cells if rr == r)
                if row_len > 0:
                    new_shape.append(row_len)
            if sum(new_shape) == sum(shape) - hook_length:
                # Compute height: (#rows spanned - 1)
                rows_spanned = len({r for r, _ in path})
                height = rows_spanned - 1
                hooks.append((tuple(sorted(new_shape, reverse=True)), height))
            return
        
        last = path[-1]
        # Adjacent cells along rim: up or left
        for dr, dc in [(-1, 0), (0, -1)]:
            nxt = (last[0] + dr, last[1] + dc)
            if nxt in cells and nxt not in visited and is_border(nxt):
                dfs(path + [nxt], visited | {nxt})
    
    for corner in corners:
        dfs([corner], {corner})
    
    return hooks


def character_S(partition, cycle_lengths):
    """
    Compute χ_{S^λ}(σ) using the Murnaghan–Nakayama rule.
    """
    @lru_cache(None)
    def mn(shape, cycles):
        if not cycles:
            return 1
        total = 0
        c = cycles[0]
        for new_shape, height in rim_hooks(shape, c):
            total += ((-1) ** height) * mn(new_shape, cycles[1:])
        return total
    
    return mn(tuple(partition), tuple(cycle_lengths))


def main():
    parser = argparse.ArgumentParser(
        prog="character.py",
        description="Compute χ_{M^λ}(C_i) and χ_{S^λ}(C_i) for S_n, e.g. n=10."
    )
    parser.add_argument(
        "-p", "--partition",
        required=True,
        help="Comma-separated partition list, e.g. '5,3,2'"
    )
    parser.add_argument(
        "-c", "--cycles",
        required=True,
        help="Comma-separated cycle counts, e.g. '3,1,0,0,1,0,0,0,0,0'"
    )
    # -h/--help is automatic

    args = parser.parse_args()
    partition = list(map(int, args.partition.split(",")))
    cycle_counts = list(map(int, args.cycles.split(",")))

    if len(cycle_counts) != sum(partition):
        raise ValueError("Cycle counts and partition length are not valid.")

    cycle_lengths = parse_cycle_type(cycle_counts)
    chi_M = character_M(partition, cycle_lengths)
    chi_S = character_S(partition, cycle_lengths)

    print(f"chi_M^lambda(C_i) = {chi_M}")
    print(f"chi_S^lambda(C_i) = {chi_S}")

if __name__ == "__main__":
    main()
