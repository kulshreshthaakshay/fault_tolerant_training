"""
Comprehensive Validation Suite
==============================
Distributed Training System for LLM Dataset w/ Fault-Tolerant Enabled

Validates all core components WITHOUT requiring live GPUs or a real
distributed process group.  Every `torch.distributed` primitive is
mocked so tests run on any single-CPU machine.

Run:
    pytest src/tests/test_fault_tolerant_training.py -v
"""

import contextlib
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure `src/` is on sys.path so that project imports resolve correctly.
# ---------------------------------------------------------------------------
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================================
#  Fixtures
# ============================================================================

# Mock objects for transformers and datasets
mock_config = MagicMock()
mock_config.hidden_size = 768
mock_config.vocab_size = 30522 # A common BERT vocab size

class DummyTransformer(nn.Module):
    """
    A dummy transformer model that mimics the output structure of HuggingFace
    models, specifically for `pooler_output` and `last_hidden_state`.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 8

        class FakeOutputs:
            def __init__(self, batch_size, seq_len, hidden_size):
                self.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
                self.pooler_output = torch.randn(batch_size, hidden_size)
        return FakeOutputs(batch_size, seq_len, self.hidden_size)

# This patcher will return a NEW instance of DummyTransformer every time it's called
mock_automodel_patcher = patch(
    "transformers.AutoModel.from_pretrained",
    side_effect=lambda *args, **kwargs: DummyTransformer(mock_config.hidden_size)
)

mock_ds = MagicMock()
mock_ds.__len__.return_value = 1000
mock_ds.__getitem__.return_value = {"text": "sample text", "label": 0}


@pytest.fixture(autouse=True)
def _mock_dist_and_hf():
    """
    Mock ``torch.distributed`` and HuggingFace components for the entire test session.
    """
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.get_world_size", return_value=1), \
         patch("torch.distributed.barrier"), \
         patch("torch.distributed.broadcast"), \
         patch("torch.distributed.all_reduce"), \
         patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), \
         mock_automodel_patcher, \
         patch("transformers.AutoTokenizer.from_pretrained"), \
         patch("datasets.load_dataset", return_value=mock_ds):
        yield


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Provide a clean, temporary checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return str(ckpt_dir)


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Provide a clean, temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def sample_model():
    """A lightweight stand-in model (no HuggingFace download)."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    return model


@pytest.fixture
def sample_optimizer(sample_model):
    return torch.optim.AdamW(sample_model.parameters(), lr=2e-5)


# ============================================================================
#  1.  MODEL — TransformerModel
# ============================================================================

class TestTransformerModel:
    """Validate TransformerModel architecture and forward pass."""

    @pytest.fixture(autouse=True)
    def _import_model(self):
        from models.model import TransformerModel
        self.TransformerModel = TransformerModel

    # ---- Architecture --------------------------------------------------

    def test_model_instantiation(self):
        """Model builds without error with default config."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2
        )
        assert model is not None
        assert hasattr(model, "transformer")
        assert hasattr(model, "classifier")
        assert hasattr(model, "dropout")
        assert hasattr(model, "pooler")

    def test_classifier_output_shape(self):
        """Classifier head has correct output dimensions."""
        for n_cls in (2, 5, 10):
            model = self.TransformerModel(
                model_name="bert-base-uncased", num_classes=n_cls
            )
            assert model.classifier.out_features == n_cls

    def test_classifier_weight_init(self):
        """Classifier weights use N(0, 0.02) init, bias zeros."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2
        )
        assert model.classifier.bias.abs().sum().item() == 0.0
        # Std should be roughly 0.02 (with some variance for small tensors)
        std = model.classifier.weight.std().item()
        assert 0.005 < std < 0.06, f"Unexpected std: {std}"

    # ---- Forward pass --------------------------------------------------

    def test_forward_output_shape(self):
        """Forward pass returns logits of shape (batch, num_classes)."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2
        )
        model.eval()
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        assert logits.shape == (batch_size, 2)

    def test_no_double_pooling_for_bert(self):
        """BERT pooler_output should NOT pass through self.pooler."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2
        )
        model.eval()
        # Hook into pooler to verify it is NOT called
        pooler_called = {"called": False}
        original_forward = model.pooler.forward

        def spy_pooler(x):
            pooler_called["called"] = True
            return original_forward(x)

        model.pooler.forward = spy_pooler
        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        assert not pooler_called["called"], (
            "self.pooler was called on BERT pooler_output — double pooling detected!"
        )

    def test_pooler_used_for_mean_pooling_fallback(self):
        """When pooler_output is absent, self.pooler IS used."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2
        )
        model.eval()

        # Patch the transformer to return output WITHOUT pooler_output
        batch_size, seq_len = 2, 8
        hidden_size = model.config.hidden_size
        
        # Create a custom DummyTransformer that explicitly returns pooler_output = None
        class DummyTransformerWithoutPooler(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_len = input_ids.shape[1] if input_ids is not None else 8

                class FakeOutputsWithoutPooler:
                    def __init__(self, batch_size, seq_len, hidden_size):
                        self.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
                        self.pooler_output = None
                return FakeOutputsWithoutPooler(batch_size, seq_len, self.hidden_size)
                
        model.transformer = DummyTransformerWithoutPooler(hidden_size)

        pooler_called = {"called": False}
        original_forward = model.pooler.forward

        def spy_pooler(x):
            pooler_called["called"] = True
            return original_forward(x)

        model.pooler.forward = spy_pooler

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        assert pooler_called["called"], (
            "self.pooler was NOT called on mean-pooled output — "
            "fallback path skipped pooler!"
        )
        assert logits.shape == (batch_size, 2)

    def test_dropout_is_applied(self):
        """Dropout layer is present and has correct rate."""
        model = self.TransformerModel(
            model_name="bert-base-uncased", num_classes=2, dropout_rate=0.3
        )
        assert model.dropout.p == 0.3


# ============================================================================
#  2.  DATA LOADER — TextDataset
# ============================================================================

class TestTextDataset:
    """Validate TextDataset tokenisation and __getitem__."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.data_loader import TextDataset
        self.TextDataset = TextDataset

    def test_dataset_length(self):
        texts = ["hello world", "foo bar baz"]
        labels = [0, 1]
        ds = self.TextDataset(texts, labels, max_length=32)
        assert len(ds) == 2

    def test_getitem_keys(self):
        texts = ["sample text"]
        labels = [1]
        ds = self.TextDataset(texts, labels, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_shapes(self):
        texts = ["sample text for shape test"]
        labels = [0]
        max_len = 64
        ds = self.TextDataset(texts, labels, max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        assert item["labels"].shape == ()  # scalar

    def test_label_dtype(self):
        texts = ["test"]
        labels = [1]
        ds = self.TextDataset(texts, labels, max_length=16)
        assert ds[0]["labels"].dtype == torch.long

    def test_lazy_logger_init(self):
        """Logger in data_loader should NOT be initialised at import time."""
        import utils.data_loader as dl_module
        # Reset logger to simulate fresh import
        dl_module.logger = None
        # At this point logger should be None
        assert dl_module.logger is None
        # After creating a TextDataset, logger should be initialized
        ds = self.TextDataset(["test"], [0], max_length=16)
        assert dl_module.logger is not None


# ============================================================================
#  3.  DISTRIBUTED LOGGER
# ============================================================================

class TestDistributedLogger:
    """Validate DistributedLogger configuration and behavior."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.logger import DistributedLogger
        self.DistributedLogger = DistributedLogger

    def test_logger_creates_log_dir(self, tmp_log_dir):
        log_subdir = os.path.join(tmp_log_dir, "nested")
        dl = self.DistributedLogger("test_create", log_dir=log_subdir)
        assert os.path.isdir(log_subdir)

    def test_logger_rank_when_dist_initialized(self, tmp_log_dir):
        """When dist is initialised, rank/world_size come from dist."""
        with patch("torch.distributed.get_rank", return_value=3), \
             patch("torch.distributed.get_world_size", return_value=8):
            dl = self.DistributedLogger("test_rank", log_dir=tmp_log_dir)
            assert dl.rank == 3
            assert dl.world_size == 8

    def test_logger_level_is_info(self, tmp_log_dir):
        dl = self.DistributedLogger("test_level", log_dir=tmp_log_dir)
        assert dl.logger.level == logging.INFO

    def test_console_handler_only_on_rank_0(self, tmp_log_dir):
        """Console (stream) handler is only attached to rank 0."""
        with patch("torch.distributed.get_rank", return_value=0):
            dl0 = self.DistributedLogger("test_console_r0", log_dir=tmp_log_dir)
            stream_handlers = [
                h for h in dl0.logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            assert len(stream_handlers) == 1

        with patch("torch.distributed.get_rank", return_value=1):
            dl1 = self.DistributedLogger("test_console_r1", log_dir=tmp_log_dir)
            stream_handlers = [
                h for h in dl1.logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            assert len(stream_handlers) == 0

    def test_file_handler_exists(self, tmp_log_dir):
        dl = self.DistributedLogger("test_fh", log_dir=tmp_log_dir)
        file_handlers = [
            h for h in dl.logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1
        # File should be rank-specific
        assert any("rank_0" in h.baseFilename for h in file_handlers)

    # ---- Metrics logger ------------------------------------------------

    def test_metrics_logger_is_isolated(self, tmp_log_dir):
        """metrics_logger should be a separate logger, not self.logger."""
        dl = self.DistributedLogger("test_iso", log_dir=tmp_log_dir)
        assert dl.metrics_logger is not dl.logger
        assert dl.metrics_logger.name != dl.logger.name

    def test_metrics_logger_level_info(self, tmp_log_dir):
        dl = self.DistributedLogger("test_ml_level", log_dir=tmp_log_dir)
        assert dl.metrics_logger.level == logging.INFO

    def test_metrics_logger_propagate_false(self, tmp_log_dir):
        """metrics_logger should NOT propagate to root (no console spam)."""
        dl = self.DistributedLogger("test_prop", log_dir=tmp_log_dir)
        assert dl.metrics_logger.propagate is False

    def test_metrics_handler_level_info(self, tmp_log_dir):
        dl = self.DistributedLogger("test_mh_level", log_dir=tmp_log_dir)
        for h in dl.metrics_logger.handlers:
            if isinstance(h, logging.FileHandler):
                assert h.level == logging.INFO

    def test_log_metrics_writes_json(self, tmp_log_dir):
        dl = self.DistributedLogger("test_json", log_dir=tmp_log_dir)
        dl.log_metrics({"loss": 0.42, "epoch": 1})
        # Flush handlers
        for h in dl.metrics_logger.handlers:
            h.flush()
        metrics_file = os.path.join(tmp_log_dir, "metrics_rank_0.json")
        assert os.path.exists(metrics_file)
        with open(metrics_file) as f:
            content = f.read().strip()
        data = json.loads(content)
        assert data["loss"] == 0.42
        assert "timestamp" in data
        assert data["rank"] == 0

    def test_text_logs_not_in_metrics_file(self, tmp_log_dir):
        """Standard info/warning/error text should NOT appear in .json file."""
        dl = self.DistributedLogger("test_leak", log_dir=tmp_log_dir)
        dl.info("This is a text log")
        dl.warning("This is a warning")
        for h in dl.metrics_logger.handlers:
            h.flush()
        for h in dl.logger.handlers:
            h.flush()
        metrics_file = os.path.join(tmp_log_dir, "metrics_rank_0.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                content = f.read()
            assert "This is a text log" not in content
            assert "This is a warning" not in content

    def test_logger_not_initialized_at_import_in_train(self):
        """train.py should have logger = None at module level."""
        import train as train_module
        # We check the SOURCE, not runtime (since tests may have set it)
        import inspect
        src = inspect.getsource(train_module)
        assert "logger = None" in src


# ============================================================================
#  4.  CHECKPOINT — Atomic Save / Load / Cleanup
# ============================================================================

class TestCheckpointing:
    """Validate fault-tolerant checkpoint save, load, and cleanup."""

    @pytest.fixture(autouse=True)
    def _import(self):
        # We need the logger to exist for save_checkpoint_atomic
        import train as train_module
        from utils.logger import DistributedLogger
        if train_module.logger is None:
            train_module.logger = DistributedLogger("test_train", log_dir=tempfile.mkdtemp())
        self.train_module = train_module

    def test_save_checkpoint_creates_file(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        self.train_module.save_checkpoint_atomic(
            sample_model, sample_optimizer, epoch=1, batch_idx=0,
            loss=0.5, checkpoint_dir=tmp_checkpoint_dir,
        )
        files = glob.glob(os.path.join(tmp_checkpoint_dir, "model_epoch_*.pth"))
        assert len(files) == 1

    def test_checkpoint_contains_required_keys(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        self.train_module.save_checkpoint_atomic(
            sample_model, sample_optimizer, epoch=2, batch_idx=10,
            loss=0.3, checkpoint_dir=tmp_checkpoint_dir,
        )
        ckpt_path = glob.glob(os.path.join(tmp_checkpoint_dir, "*.pth"))[0]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "batch_idx", "model_state_dict",
                     "optimizer_state_dict", "loss", "world_size", "timestamp"):
            assert key in ckpt, f"Missing key: {key}"
        assert ckpt["epoch"] == 2
        assert ckpt["batch_idx"] == 10

    def test_checkpoint_unwraps_ddp_module(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        """Checkpoint should save unwrapped model keys (no 'module.' prefix)."""
        # Simulate DDP wrapper
        wrapped = MagicMock()
        wrapped.module = sample_model
        wrapped.state_dict = lambda: {
            f"module.{k}": v for k, v in sample_model.state_dict().items()
        }
        self.train_module.save_checkpoint_atomic(
            wrapped, sample_optimizer, epoch=1, batch_idx=0,
            loss=0.1, checkpoint_dir=tmp_checkpoint_dir,
        )
        ckpt_path = glob.glob(os.path.join(tmp_checkpoint_dir, "*.pth"))[0]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ckpt["model_state_dict"]:
            assert not key.startswith("module."), (
                f"DDP 'module.' prefix leaked into checkpoint: {key}"
            )

    def test_checkpoint_atomic_naming(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        self.train_module.save_checkpoint_atomic(
            sample_model, sample_optimizer, epoch=3, batch_idx=200,
            loss=0.2, checkpoint_dir=tmp_checkpoint_dir,
        )
        expected = os.path.join(tmp_checkpoint_dir, "model_epoch_3_batch_200.pth")
        assert os.path.exists(expected)

    def test_load_checkpoint_restores_state(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        # Save first
        original_state = {k: v.clone() for k, v in sample_model.state_dict().items()}
        self.train_module.save_checkpoint_atomic(
            sample_model, sample_optimizer, epoch=5, batch_idx=0,
            loss=0.1, checkpoint_dir=tmp_checkpoint_dir,
        )
        # Perturb weights
        with torch.no_grad():
            for p in sample_model.parameters():
                p.add_(torch.randn_like(p))
        # Load
        epoch, batch = self.train_module.load_checkpoint(
            sample_model, sample_optimizer, checkpoint_dir=tmp_checkpoint_dir,
        )
        assert epoch == 5
        assert batch == 0
        for name, param in sample_model.named_parameters():
            assert torch.allclose(param, original_state[name]), (
                f"Parameter {name} not restored correctly"
            )

    def test_load_checkpoint_no_files(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        epoch, batch = self.train_module.load_checkpoint(
            sample_model, sample_optimizer, checkpoint_dir=tmp_checkpoint_dir,
        )
        assert epoch == 0
        assert batch == 0

    def test_cleanup_old_checkpoints(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        """Only the last N checkpoints should be kept."""
        for i in range(5):
            path = os.path.join(tmp_checkpoint_dir, f"model_epoch_{i}_batch_0.pth")
            torch.save({"dummy": i}, path)
            time.sleep(0.05)  # Ensure different mtime
        self.train_module.cleanup_old_checkpoints(tmp_checkpoint_dir, keep_last=2)
        remaining = glob.glob(os.path.join(tmp_checkpoint_dir, "model_epoch_*.pth"))
        assert len(remaining) == 2

    def test_non_rank0_skips_save(self, sample_model, sample_optimizer, tmp_checkpoint_dir):
        """Non-rank-0 processes should NOT write checkpoints."""
        with patch("torch.distributed.get_rank", return_value=1):
            self.train_module.save_checkpoint_atomic(
                sample_model, sample_optimizer, epoch=1, batch_idx=0,
                loss=0.5, checkpoint_dir=tmp_checkpoint_dir,
            )
        files = glob.glob(os.path.join(tmp_checkpoint_dir, "*.pth"))
        assert len(files) == 0


# ============================================================================
#  5.  TRAINING LOOP — train_distributed
# ============================================================================

class TestTrainDistributed:
    """Validate training loop behavior with mocked components."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import train as train_module
        from utils.logger import DistributedLogger
        if train_module.logger is None:
            train_module.logger = DistributedLogger("test_train_loop", log_dir=tempfile.mkdtemp())
        self.train_module = train_module

    def _make_fake_loader(self, n_batches=8, seq_len=16, batch_size=4):
        """Create a fake DataLoader yielding random token batches."""
        batches = []
        for _ in range(n_batches):
            batches.append({
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "labels": torch.randint(0, 2, (batch_size,)),
            })
        return batches

    def test_gradient_accumulation_steps_param(self):
        """train_distributed accepts gradient_accumulation_steps."""
        import inspect
        sig = inspect.signature(self.train_module.train_distributed)
        assert "gradient_accumulation_steps" in sig.parameters

    def test_scheduler_param(self):
        """train_distributed accepts a scheduler."""
        import inspect
        sig = inspect.signature(self.train_module.train_distributed)
        assert "scheduler" in sig.parameters

    def test_debug_flag_param(self):
        """train_distributed accepts a debug flag."""
        import inspect
        sig = inspect.signature(self.train_module.train_distributed)
        assert "debug" in sig.parameters
        # Default should be False
        assert sig.parameters["debug"].default is False


# ============================================================================
#  6.  MAIN FUNCTION STRUCTURE
# ============================================================================

class TestMainStructure:
    """Validate the ordering and presence of key operations in main()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import train as train_module
        self.train_module = train_module
        import inspect
        self.main_source = inspect.getsource(train_module.main)

    def test_dist_init_before_logger(self):
        """dist.init_process_group should come before DistributedLogger."""
        src = self.main_source
        init_pos = src.find("dist.init_process_group")
        logger_pos = src.find("DistributedLogger")
        assert init_pos < logger_pos, (
            "DistributedLogger is created before dist.init_process_group!"
        )

    def test_checkpoint_load_before_ddp(self):
        """load_checkpoint should be called before DDP wrapping."""
        src = self.main_source
        load_pos = src.find("load_checkpoint")
        ddp_pos = src.find("DDP(")
        assert load_pos < ddp_pos, (
            "load_checkpoint is called after DDP wrapping!"
        )

    def test_broadcast_after_checkpoint_load(self):
        """dist.broadcast should sync resume info after load_checkpoint."""
        src = self.main_source
        load_pos = src.find("load_checkpoint")
        broadcast_pos = src.find("dist.broadcast")
        assert broadcast_pos > load_pos, (
            "dist.broadcast is not present after load_checkpoint!"
        )

    def test_ddp_after_broadcast(self):
        """DDP wrapping should happen after the broadcast."""
        src = self.main_source
        broadcast_pos = src.find("dist.broadcast")
        ddp_pos = src.find("DDP(")
        assert ddp_pos > broadcast_pos, (
            "DDP wrapping happens before dist.broadcast!"
        )

    def test_prometheus_after_init(self):
        """Prometheus start_http_server should be after model/data init."""
        src = self.main_source
        prom_pos = src.find("start_http_server")
        dataset_pos = src.find("TextDataset")
        assert prom_pos > dataset_pos, (
            "Prometheus starts before dataset initialization!"
        )

    def test_prometheus_has_oserror_handling(self):
        """Prometheus start should be wrapped in try/except OSError."""
        src = self.main_source
        assert "except OSError" in src, (
            "Prometheus start_http_server is not wrapped in try/except OSError!"
        )

    def test_scheduler_created(self):
        """A learning rate scheduler should be created in main."""
        src = self.main_source
        assert "get_linear_schedule_with_warmup" in src, (
            "No LR scheduler (get_linear_schedule_with_warmup) in main!"
        )

    def test_scheduler_passed_to_train(self):
        """Scheduler should be passed to train_distributed."""
        src = self.main_source
        assert "scheduler=scheduler" in src or "scheduler = scheduler" in src, (
            "Scheduler not passed to train_distributed!"
        )

    def test_gradient_accumulation_passed(self):
        """gradient_accumulation_steps should be passed to train_distributed."""
        src = self.main_source
        assert "gradient_accumulation_steps=" in src

    def test_module_level_logger_is_none(self):
        """train.py module-level logger should be None (lazy init)."""
        import inspect
        full_src = inspect.getsource(self.train_module)
        # Should contain the assignment at module scope
        assert "\nlogger = None\n" in full_src or full_src.startswith("logger = None")


# ============================================================================
#  7.  ANOMALY HANDLING
# ============================================================================

class TestAnomalyHandling:
    """Validate the training anomaly detection and recovery."""

    def test_detect_anomaly_off_by_default(self):
        """detect_anomaly should NOT run when debug=False."""
        import inspect
        import train as train_module
        src = inspect.getsource(train_module.train_distributed)
        # Should use contextlib.nullcontext when not debugging
        assert "contextlib.nullcontext()" in src
        assert "if debug" in src

    def test_nan_loss_detection_in_source(self):
        """Training loop should check for NaN/Inf losses."""
        import inspect
        import train as train_module
        src = inspect.getsource(train_module.train_distributed)
        assert "torch.isnan(loss)" in src
        assert "torch.isinf(loss)" in src

    def test_gradient_clipping_in_source(self):
        """clip_grad_norm_ should be called with max_norm=1.0."""
        import inspect
        import train as train_module
        src = inspect.getsource(train_module.train_distributed)
        assert "clip_grad_norm_" in src
        assert "max_norm=1.0" in src


# ============================================================================
#  8.  INTEGRATION — End-to-End Smoke Test (mocked dist)
# ============================================================================

class TestIntegrationSmoke:
    """
    Lightweight integration test that verifies components work together
    without live GPUs by mocking CUDA operations.
    """

    def test_model_forward_backward(self):
        """Full forward + backward pass on CPU with TransformerModel."""
        from models.model import TransformerModel
        model = TransformerModel(model_name="bert-base-uncased", num_classes=2)
        model.train()
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,))

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Verify gradients flow
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients detected after backward pass!"

    def test_dataset_to_dataloader(self):
        """TextDataset works with a standard DataLoader."""
        from utils.data_loader import TextDataset
        texts = ["hello world", "testing data loader", "third sample"]
        labels = [0, 1, 0]
        ds = TextDataset(texts, labels, max_length=32)
        dl = DataLoader(ds, batch_size=2, shuffle=False)

        batch = next(iter(dl))
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

    def test_checkpoint_round_trip(self, tmp_checkpoint_dir):
        """Save → perturb → load → verify weights match."""
        import train as train_module
        from utils.logger import DistributedLogger
        if train_module.logger is None:
            train_module.logger = DistributedLogger("test_rt", log_dir=tempfile.mkdtemp())

        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters())

        # Snapshot
        original = {k: v.clone() for k, v in model.state_dict().items()}

        # Save
        train_module.save_checkpoint_atomic(
            model, optimizer, epoch=1, batch_idx=0, loss=0.5,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        # Perturb
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Load
        train_module.load_checkpoint(
            model, optimizer, checkpoint_dir=tmp_checkpoint_dir,
        )

        # Assert
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original[k]), f"{k} not restored"
