def build_model(input_dim: int, cfg: dict = None): raise NotImplementedError
def get_weights(model) -> list:    # list of numpy arrays, for Flower
    raise NotImplementedError
def set_weights(model, weights: list) -> None:
    raise NotImplementedError
def train(model, X, y, epochs, lr, batch_size, **kwargs):  # returns model
    raise NotImplementedError