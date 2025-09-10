import numpy as np 
import os 

class DeckStack: 
    def __init__(self, num_decks=1_000_000, batch_size=10_000, out_dir="decks", seed=12):
        self.num_decks = num_decks
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.master_rng = np.random.default_rng(seed) 
        os.makedirs(out_dir, exist_ok=True)
    
    def generate_and_store_decks(self):
        """
        Generate and store shuffled decks of 52 cards with 26 red (0) and 26 black (1) cards
        """
        deck_template = np.array([0] * 26 + [1] * 26)
        num_batches = self.num_decks // self.batch_size

        for batch_idx in range(num_batches):
            # Generate a batch of shuffled decks
            decks = np.empty((self.batch_size, 52), dtype=int)
            for i in range(self.batch_size):
                decks[i] = self.master_rng.permutation(deck_template)

            # Save the batch to a file
            batch_file = os.path.join(self.out_dir, f"decks_batch_{batch_idx + 1}.npy")
            np.save(batch_file, decks)
            print(f"Saved batch {batch_idx + 1}/{num_batches} to {batch_file}")

        print("All decks generated and stored.")

if __name__ == "__main__":
    deck_stack = DeckStack()
    deck_stack.generate_and_store_decks()


