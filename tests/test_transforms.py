import torch
from ser.transforms import flip


def test_flip():

    test_tensor = torch.tensor([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    test_result = torch.tensor([[6, 0, 0], [5, 4, 0], [3, 2, 1]])

    assert flip()(test_tensor).all() == test_result.all()
