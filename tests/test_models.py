import torch

from downstream.models.tempCNN import TemporalCNN
from models.tempCNN import TempCNN
from models.vicreg import VICRegNet


def test_tempcnn_shape_and_legacy_key_names():
    model = TempCNN(input_dim=2)
    output = model(torch.randn(4, 2, 12))
    assert output.shape == (4, 128)
    keys = model.state_dict().keys()
    assert "conv_bn_relu1.block.0.weight" in keys
    assert "conv_bn_relu2.block.0.weight" in keys
    assert "conv_bn_relu3.block.0.weight" in keys


def test_vicreg_output_shapes():
    model = VICRegNet()
    z1, z2 = model(torch.randn(4, 2, 12), torch.randn(4, 10, 12))
    assert z1.shape == (4, 256)
    assert z2.shape == (4, 256)


def test_downstream_output_shape():
    model = TemporalCNN(num_classes=2)
    logits = model(torch.randn(4, 2, 12), torch.randn(4, 10, 12))
    assert logits.shape == (4, 2)
