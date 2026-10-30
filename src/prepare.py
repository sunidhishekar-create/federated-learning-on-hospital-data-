from src.data_loader import prepare_pool_and_test
from src.partition import build_hospitals

if __name__ == "__main__":
    info = prepare_pool_and_test()
    print(f"features: {info['input_dim']} | pool: {info['n_pool']} | test: {info['n_test']}")
    print(build_hospitals().to_string(index=False))