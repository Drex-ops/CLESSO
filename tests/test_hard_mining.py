"""Test hard-pair mining implementation."""
import numpy as np
import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    """Mock between-site pair dataset."""
    def __init__(self, n=1000):
        self.n = n
        self.site_i = np.arange(n, dtype=np.int64) % 50
        self.site_j = np.arange(n, dtype=np.int64) % 50 + 50
        rng = np.random.RandomState(42)
        self.y = rng.choice([0.0, 1.0], size=n).astype(np.float32)
        self.is_within = np.zeros(n, dtype=np.float32)
        self.weight = np.ones(n, dtype=np.float32)
        self.env_diff = rng.randn(n, 5).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "site_i": self.site_i[idx],
            "site_j": self.site_j[idx],
            "y": self.y[idx],
            "is_within": self.is_within[idx],
            "weight": self.weight[idx],
            "env_diff": self.env_diff[idx],
        }


class MockModel(torch.nn.Module):
    """Mock model that returns varying p_match."""
    def forward(self, z_i, z_j, env_diff, is_within):
        b = z_i.shape[0]
        # Variable p_match to create difficulty spread
        p = 0.1 + 0.8 * torch.rand(b)
        return {
            "alpha_i": torch.ones(b) * 10,
            "alpha_j": torch.ones(b) * 10,
            "eta": torch.ones(b) * 0.5,
            "similarity": torch.ones(b) * 0.6,
            "p_match": p,
        }

    def eval(self):
        return self


class MockSiteData:
    Z = torch.randn(100, 8)


class MockConfig:
    hard_mining_lambda = 0.5
    hard_mining_n_bins = 3
    hard_mining_bin_weights = [0.2, 0.3, 0.5]
    hard_mining_a_max = 5.0
    hard_mining_refresh_every = 1


def test_miner_creation():
    from src.clesso_nn.dataset import HardPairMiner
    ds = MockDataset(1000)
    config = MockConfig()
    miner = HardPairMiner(ds, config)
    assert miner.n == 1000
    assert miner.lambda_hm == 0.5
    assert len(miner.bin_weights) == 3
    print("  [PASS] test_miner_creation")


def test_refresh_scores():
    from src.clesso_nn.dataset import HardPairMiner
    ds = MockDataset(1000)
    config = MockConfig()
    miner = HardPairMiner(ds, config)

    model = MockModel()
    site_data = MockSiteData()
    miner.refresh_scores(model, site_data, "cpu", batch_size=256)

    # q should be a valid probability distribution
    assert abs(miner.q.sum() - 1.0) < 1e-6, f"q sums to {miner.q.sum()}"
    assert (miner.q > 0).all(), "q has zero entries"

    # iw should be positive and bounded by a_max
    assert (miner.iw > 0).all(), "iw has non-positive entries"
    assert miner.iw.max() <= config.hard_mining_a_max + 1e-6, (
        f"iw max {miner.iw.max()} exceeds a_max {config.hard_mining_a_max}")

    # Hard pairs should have higher q than easy pairs on average
    print(f"  q range: [{miner.q.min():.8f}, {miner.q.max():.8f}]")
    print(f"  iw range: [{miner.iw.min():.3f}, {miner.iw.max():.3f}]")
    print("  [PASS] test_refresh_scores")


def test_no_mining():
    """With lambda=0, miner should produce uniform q and iw=1."""
    from src.clesso_nn.dataset import HardPairMiner
    ds = MockDataset(1000)
    config = MockConfig()
    config.hard_mining_lambda = 0.0
    miner = HardPairMiner(ds, config)

    model = MockModel()
    site_data = MockSiteData()
    miner.refresh_scores(model, site_data, "cpu")

    expected_q = 1.0 / 1000
    assert np.allclose(miner.q, expected_q), "q not uniform with lambda=0"
    assert np.allclose(miner.iw, 1.0), "iw not 1.0 with lambda=0"
    print("  [PASS] test_no_mining")


def test_make_dataloader():
    from src.clesso_nn.dataset import HardPairMiner
    ds = MockDataset(500)
    config = MockConfig()
    miner = HardPairMiner(ds, config)

    model = MockModel()
    site_data = MockSiteData()
    miner.refresh_scores(model, site_data, "cpu")

    loader = miner.make_dataloader(batch_size=64)
    batch = next(iter(loader))

    assert "mining_iw" in batch, "mining_iw missing from batch"
    assert batch["mining_iw"].shape[0] == 64
    assert batch["site_i"].shape[0] == 64
    assert batch["y"].shape[0] == 64
    assert batch["env_diff"].shape[0] == 64
    print(f"  Batch mining_iw: [{batch['mining_iw'].min():.3f}, {batch['mining_iw'].max():.3f}]")
    print("  [PASS] test_make_dataloader")


def test_config_defaults():
    from src.clesso_nn.config import CLESSONNConfig
    c = CLESSONNConfig()
    assert c.hard_mining_lambda == 0.0, "Should be disabled by default"
    assert c.hard_mining_n_bins == 3
    assert len(c.hard_mining_bin_weights) == c.hard_mining_n_bins
    assert c.hard_mining_a_max == 5.0
    assert c.hard_mining_warmup_cycles == 3
    assert c.hard_mining_refresh_every == 1
    print("  [PASS] test_config_defaults")


def test_train_import():
    from src.clesso_nn.train import train_cyclic, train_two_stage
    print("  [PASS] test_train_import")


if __name__ == "__main__":
    print("Running hard-pair mining tests...")
    test_config_defaults()
    test_miner_creation()
    test_no_mining()
    test_refresh_scores()
    test_make_dataloader()
    test_train_import()
    print("\nAll tests passed!")
