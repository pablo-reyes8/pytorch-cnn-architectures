# tests/test_serialization.py
import torch
import tempfile
import os

from model.compuder_scaler import * 
from model.Efficent_Net import *

def test_save_and_load_state_dict_tmpfile():
    model = EfficientNet(num_classes=13, scaler=CompoundScaler(phi=1))
    sd = model.state_dict()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        torch.save(sd, path)
        assert os.path.exists(path)

        model2 = EfficientNet(num_classes=13, scaler=CompoundScaler(phi=1))
        model2.load_state_dict(torch.load(path, map_location="cpu"))
        # compara algunos pesos
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2)
