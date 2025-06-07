import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, processed_input):
    return model.predict(processed_input)[0]
