from super_mario_diffusion.model import Model
from super_mario_diffusion.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
