import pickle

with open("best_hmm_model_latest.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)
