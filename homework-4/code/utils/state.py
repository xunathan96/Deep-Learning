# Contains all helper functions related to saving/loading states
import os
import pickle
import torch

def save_model(model, config):
    MODEL_PATH = config.model.path + config.model.name + ".pth"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    #torch.save(model, MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH)

def load_model(config, model):
    MODEL_PATH = config.model.path + config.model.name + ".pth"
    DEVICE = config.device
    #model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Remember that you must call model.eval() to set dropout and
    # batch normalization layers to evaluation mode before running inference.
    model.to(DEVICE)
    model.eval()
    return model

def save_statistics(loss, accuracy, config):
    PATH_LOSS = config.model.path + config.model.name + "_loss.pkl"
    PATH_ACC = config.model.path + config.model.name + "_accuracy.pkl"
    
    os.makedirs(os.path.dirname(PATH_LOSS), exist_ok=True)
    
    f = open(PATH_LOSS, "wb")
    pickle.dump(loss, f)
    f.close()
    
    f = open(PATH_ACC, "wb")
    pickle.dump(accuracy, f)
    f.close()

def load_statistics(config):
    PATH_LOSS = config.model.path + config.model.name + "_loss.pkl"
    PATH_ACC = config.model.path + config.model.name + "_accuracy.pkl"

    f = open(PATH_LOSS, "rb")
    loss = pickle.load(f)
    f.close()

    f = open(PATH_ACC, "rb")
    accuracy = pickle.load(f)
    f.close()

    return loss, accuracy


