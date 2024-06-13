from tqdm import tqdm

class ProgressBar:
    def __init__(self, total, desc='Processing'):
        self.pbar = tqdm(total=total, desc=desc)

    def update(self, n=1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()