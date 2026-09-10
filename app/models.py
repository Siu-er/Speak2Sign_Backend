"""Model lifecycle: load Whisper, the ASL glosser and the SiGML generator once
and hold them for the request handlers to use."""

import logging
import time

import torch

from app import config
from app.services.glosser import ASLGlosser
from app.services.sigml import SiGMLGenerator

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_processor = None
        self.whisper_model = None
        self.glosser = None
        self.sigml = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info(f"Loading Whisper ({config.WHISPER_MODEL_NAME}) on {self.device}")
        start = time.time()
        self.whisper_processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            config.WHISPER_MODEL_NAME
        ).to(self.device)
        self.whisper_model.eval()
        if hasattr(torch, "compile"):
            try:
                self.whisper_model = torch.compile(self.whisper_model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without: {e}")
        logger.info(f"Whisper loaded in {time.time() - start:.2f}s")

        start = time.time()
        self.glosser = ASLGlosser(config.DATA_DIR)
        logger.info(f"ASL glosser loaded in {time.time() - start:.2f}s")

        start = time.time()
        self.sigml = SiGMLGenerator(config.SIGNS_DIR)
        logger.info(f"SiGML generator loaded in {time.time() - start:.2f}s")

        self._loaded = True
        logger.info(f"Models ready on {self.device}")

    def status(self):
        return {
            "whisper_processor": self.whisper_processor is not None,
            "whisper_model": self.whisper_model is not None,
            "asl_glosser": self.glosser is not None,
            "sigml_generator": self.sigml is not None,
        }

    def clear_gpu_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


models = ModelRegistry()
