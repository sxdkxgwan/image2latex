from utils.dataset import Dataset



if __name__ == "__main__":
    path = "../data/val_filter.lst"
    myset = Dataset(path)
    for element in myset:
        print element