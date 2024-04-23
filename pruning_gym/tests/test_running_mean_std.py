import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch as th
import numpy as np
from ..running_mean_std import RunningMeanStd
from typing import Tuple
import pytest

@pytest.fixture
def running_mean_std():
    return RunningMeanStd(shape=(1, ))



@pytest.mark.parametrize("arr", [th.tensor([[1, 2, 3, 4, 5, 6],
                                           [7, 8, 9, 10, 11, 12]]).reshape(2, -1).double(),])
def test_update(running_mean_std, arr):
    #make a 4D numpy array
    assert running_mean_std.mean.shape == (1, )
    assert running_mean_std.var.shape == (1, )
    assert running_mean_std.count == 0.0001
    assert running_mean_std.mean == 0
    assert running_mean_std.var == 1

    running_mean_std.update(arr)
    assert np.isclose(running_mean_std.mean.item(), th.mean(arr).item(), atol=1e-2)
    assert np.isclose(running_mean_std.var.item(), th.var(arr).item(), atol=1e-2)
    running_mean_std.update(arr[0].reshape(1, -1))
    assert np.isclose(running_mean_std.mean.item(), 5.5, atol=1e-2)
    assert np.isclose(running_mean_std.var.item(), (th.var(arr)*2+th.var(arr[0])*1+9*2*1/3).item()/3, atol=1e-2)




