import pytest
import torch
from unittest.mock import MagicMock, patch

from app.ml.models.fusion_model import TradingFusionModel


class TestModelReproducibility:
    @pytest.fixture
    def model(self):
        config = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "d_ff": 128,
            "price_features": 39,
            "sentiment_features": 4,
            "seq_len": 30,
            "num_classes": 3,
            "reg_targets": 5,
        }
        return TradingFusionModel(**config)

    def test_model_forward_shapes(self, model):
        batch_size = 4
        seq_len = 30
        price_features = 39
        sentiment_features = 4

        x_price = torch.randn(batch_size, seq_len, price_features)
        x_sent = torch.randn(batch_size, sentiment_features)

        logits, regression = model(x_price, x_sent)

        assert logits.shape == (batch_size, 3)
        assert regression.shape == (batch_size, 5)

    def test_model_train_eval_mode(self, model):
        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

    def test_model_parameter_count(self, model):
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_model_deterministic(self, model):
        x_price = torch.randn(2, 30, 39)
        x_sent = torch.randn(2, 4)

        torch.manual_seed(42)
        out1 = model(x_price, x_sent)

        torch.manual_seed(42)
        out2 = model(x_price, x_sent)

        assert torch.allclose(out1[0], out2[0], atol=1e-6)
        assert torch.allclose(out1[1], out2[1], atol=1e-6)
