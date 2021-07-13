import os, pickle

class Dataloader():

    def dump_data(self, path, dataset):
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)