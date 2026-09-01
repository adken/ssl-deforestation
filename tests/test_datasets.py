import numpy as np

from downstream.utils.dataset import TimeSeriesDataset as DownstreamDataset
from utils.dataset import TimeSeriesDataset as PretrainDataset


def test_pretrain_dataset_is_paired(tmp_path):
    s1 = np.random.randn(5, 12, 2).astype("float32")
    s2 = (np.random.rand(5, 12, 10) * 10000).astype("float32")
    np.save(tmp_path / "s1_stack.npy", s1)
    np.save(tmp_path / "s2_stack.npy", s2)

    dataset = PretrainDataset(str(tmp_path))
    sample_s1, sample_s2 = dataset[3]

    assert len(dataset) == 5
    assert sample_s1.shape == (2, 12)
    assert sample_s2.shape == (10, 12)


def test_downstream_dataset_shapes(tmp_path):
    s1 = np.random.randn(6, 12, 2).astype("float32")
    s2 = (np.random.rand(6, 12, 10) * 10000).astype("float32")
    labels = np.array([0, 1, 0, 1, 0, 1], dtype="int64")
    np.save(tmp_path / "s1.npy", s1)
    np.save(tmp_path / "s2.npy", s2)
    np.save(tmp_path / "labels.npy", labels)

    dataset = DownstreamDataset(str(tmp_path))
    sample_s1, sample_s2, label = dataset[0]

    assert len(dataset) == 6
    assert sample_s1.shape == (2, 12)
    assert sample_s2.shape == (10, 12)
    assert label.ndim == 0
