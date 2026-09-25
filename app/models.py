"""Model lifecycle: load the ASL glosser and SiGML generator once and hold
them for the request handlers. Speech recognition runs in the browser, so no
acoustic model is loaded here."""

import logging
import time

from app import config
from app.services.glosser import ASLGlosser
from app.services.sigml import SiGMLGenerator

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.glosser = None
        self.sigml = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        start = time.time()
        self.glosser = ASLGlosser(config.DATA_DIR)
        logger.info(f"ASL glosser loaded in {time.time() - start:.2f}s")

        start = time.time()
        self.sigml = SiGMLGenerator(config.SIGNS_DIR)
        logger.info(f"SiGML generator loaded in {time.time() - start:.2f}s")

        self._loaded = True
        logger.info("Models ready")

    def status(self):
        return {
            "asl_glosser": self.glosser is not None,
            "sigml_generator": self.sigml is not None,
        }

    def clear_gpu_cache(self):
        pass  # no GPU model loaded server-side


models = ModelRegistry()
