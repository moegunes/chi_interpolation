def write_dict(dictionary, name):
    import pickle

    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(dictionary, f)
        print(f"Written dictionary with {name}.pkl")


def load_dict(name):
    import pickle

    with open(f"{name}.pkl", "rb") as f:
        return pickle.load(f)
