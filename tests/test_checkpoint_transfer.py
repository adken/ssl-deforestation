import torch

from downstream.models.tempCNN import TemporalCNN
from models.vicreg import VICRegNet


def test_vicreg_checkpoint_transfers_encoders(tmp_path):
    pretrained = VICRegNet()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": pretrained.state_dict()}, checkpoint)

    downstream = TemporalCNN()
    downstream.load_pretrained_encoders(str(checkpoint))

    assert torch.equal(
        downstream.encoder_s1.conv_bn_relu1.block[0].weight,
        pretrained.encoder_s1.conv_bn_relu1.block[0].weight,
    )
    assert torch.equal(
        downstream.encoder_s2.conv_bn_relu1.block[0].weight,
        pretrained.encoder_s2.conv_bn_relu1.block[0].weight,
    )


def test_freeze_keeps_encoders_eval_and_head_trainable():
    model = TemporalCNN()
    model.freeze_encoders()
    model.train()

    assert not model.encoder_s1.training
    assert not model.encoder_s2.training
    assert model.head.training
    assert model.classifier.training
    assert all(not p.requires_grad for p in model.encoder_s1.parameters())
    assert all(not p.requires_grad for p in model.encoder_s2.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
