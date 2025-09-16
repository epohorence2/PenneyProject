import numpy as np
import os 
import time 
from main import DeckStack



def test_generate_and_store_decks():
    # Test parameters
    num_decks = 2_000_000
    batch_size = 10_000
    out_dir = "test_decks"
    seed = 42

    # Create a DeckStack instance
    deck_stack = DeckStack(num_decks=num_decks, batch_size=batch_size, out_dir=out_dir, seed=seed)

    # Measure the time taken to generate and store decks
    start_time = time.time()
    deck_stack.generate_and_store_decks()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to generate and store {num_decks} decks: {elapsed_time:.2f} seconds")

    # Check if the correct number of batch files is created
    batch_files = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    assert len(batch_files) == num_decks // batch_size, "Incorrect number of batch files generated."

    # Load a batch file and verify its contents
    sample_batch_file = os.path.join(out_dir, batch_files[0])
    decks = np.load(sample_batch_file)

    # Check the shape of the decks array
    assert decks.shape == (batch_size, 52), "Deck batch shape is incorrect."

    # Check that each deck contains 26 red (0) and 26 black (1) cards
    for deck in decks:
        assert np.sum(deck == 0) == 26, "Deck does not contain 26 red cards."
        assert np.sum(deck == 1) == 26, "Deck does not contain 26 black cards."

    # Clean up test files
    for f in batch_files:
        os.remove(os.path.join(out_dir, f))
    os.rmdir(out_dir)

    print("All tests passed!")

if __name__ == "__main__":
    test_generate_and_store_decks()