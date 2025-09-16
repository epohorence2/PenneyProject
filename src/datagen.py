import numpy as np
import os
from helpers import PATH_DATA, debugger_factory 

HALF_DECK_SIZE = 26

@debugger_factory(show_args=True)
def get_decks(n_decks: int, seed: int, half_deck_size: int = HALF_DECK_SIZE) -> np.ndarray:
    """
    Efficiently generate `n_decks` shuffled decks using NumPy.
    
    Args:
        n_decks (int): Number of decks to generate.
        seed (int): Random seed for reproducibility.
        half_deck_size (int): Number of cards of each color in a half deck.
    
    Returns:
        np.ndarray: 2D array of shape (n_decks, num_cards), each row is a shuffled deck.
    """
    init_deck = [0] * half_deck_size + [1] * half_deck_size
    decks = np.tile(init_deck, (n_decks, 1))
    rng = np.random.default_rng(seed)
    rng.permuted(decks, axis=1, out=decks)
    return decks

@debugger_factory(show_args=True)
def save_decks_to_batches(decks: np.ndarray, batch_size: int, out_dir: str):
    """
    Save decks to batches in .npy files.
    
    Args:
        decks (np.ndarray): 2D array of decks to save.
        batch_size (int): Number of decks per batch file.
        out_dir (str): Directory to save the batch files.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_batches = len(decks) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = decks[start_idx:end_idx]

        batch_file = os.path.join(out_dir, f"decks_batch_{batch_idx + 1}.npy")
        np.save(batch_file, batch)
        print(f"Saved batch {batch_idx + 1}/{num_batches} to {batch_file}")

    print("All decks generated and stored.")

@debugger_factory(show_args=True)
def main():
    num_decks = 2_000_000
    batch_size = 10_000
    out_dir = os.path.join(PATH_DATA, "decks")
    seed = 12

    decks = get_decks(num_decks, seed)
    save_decks_to_batches(decks, batch_size, out_dir)

if __name__ == "__main__":
    main()