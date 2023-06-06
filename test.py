import numpy as np

# data = {
#     "obs": {
#         "depth": np.random.randn(224, 224, 1),
#         "state": np.random.randn(14,)
#     },
#     "reward": np.random.randn(),
#     "done": np.random.randint(2)
# }

# np.savez("0", **data)

data = np.load("0.npz", allow_pickle=True)

print(data["obs"].item()["state"].dtype)

