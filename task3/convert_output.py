import os, pickle


def save_pkl(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data



def main():
    


if __name__ == "__main__":
    main()

