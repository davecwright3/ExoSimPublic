#!/usr/bin/env python3

import pytest
import numpy as np
import matplotlib.pyplot as plt

def test_multorders():
    data_dir = "./reference/mord_test_"
    ext = ".npy"

    order1 = np.load(data_dir + "order1" + ext)
    order2 = np.load(data_dir + "order2" + ext)
    morder = np.load(data_dir + "both" + ext)


    sum_ord = np.concatenate((order1,order2),axis=1)
    diff = sum_ord - morder

    assert np.all((diff==0))
