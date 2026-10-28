def load_config(path="configs/config.yaml") -> dict: ...
def load_hospital_data(hospital_id: int):
    """Returns (X_train, y_train, X_val, y_val) as numpy arrays."""
    raise NotImplementedError
def load_global_test():
    """Returns (X_test, y_test)."""
    raise NotImplementedError
def get_input_dim() -> int:
    raise NotImplementedError