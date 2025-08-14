# jina_clip_embedding.py
import os
import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from transformers import AutoModel


# -------- Device & DType 선택 로직 --------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # (선택) MPS 지원 맥을 쓰면 아래 주석 해제
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    """
    기본값:
      - GPU: float16
      - CPU: float32
    환경변수로 재정의 가능: TORCH_DTYPE=float16|bfloat16|float32
    """
    env = (os.getenv("TORCH_DTYPE") or "").lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if env in mapping:
        return mapping[env]
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


def _device_name(dev: torch.device) -> str:
    try:
        if dev.type == "cuda":
            return torch.cuda.get_device_name(dev.index or 0)
        if dev.type == "mps":
            return "Apple MPS"
        return "CPU"
    except Exception:
        return dev.type.upper()


PRIMARY_DEVICE = _pick_device()
PRIMARY_DTYPE = _pick_dtype(PRIMARY_DEVICE)

# 약간의 성능 최적화
if PRIMARY_DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class JinaCLIPEmbedding:
    """Jina CLIP v2 model for text and image embeddings (GPU 우선, 실패 시 CPU 폴백)"""

    def __init__(self):
        self.model = None
        self.device = PRIMARY_DEVICE
        self.dtype = PRIMARY_DTYPE
        self._load_model()

    def _log(self, msg: str):
        # 필요 시 logging 모듈로 교체 가능
        print(msg)

    def _load_on(self, device: torch.device, dtype: torch.dtype):
        model_name = os.getenv("MODEL_NAME", "jinaai/jina-clip-v2")
        self._log(
            f"📦 Loading {model_name} on {device.type} "
            f"({ _device_name(device) }), dtype={dtype}"
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        return model

    def _load_model(self):
        # 1) 우선 GPU/가용 디바이스 시도
        try:
            self.model = self._load_on(self.device, self.dtype)
            self._log(
                f"✅ Model loaded on {self.device.type} "
                f"({ _device_name(self.device) })"
            )
            return
        except Exception as e:
            self._log(f"⚠️ Failed to load on {self.device.type}: {e}")

        # 2) 폴백: CPU float32
        try:
            cpu_device = torch.device("cpu")
            cpu_dtype = torch.float32
            self.model = self._load_on(cpu_device, cpu_dtype)
            self.device = cpu_device
            self.dtype = cpu_dtype
            self._log("✅ Fallback: Model loaded on CPU (float32)")
        except Exception as e:
            self._log(f"❌ Failed to load model on CPU as well: {e}")
            raise

    @torch.inference_mode()
    def encode_text(
        self, text: str, task: str = "retrieval.query"
    ) -> Optional[List[float]]:
        try:
            emb = self.model.encode_text([text], task=task)
            if isinstance(emb, torch.Tensor):
                return emb[0].detach().cpu().numpy().tolist()
            if isinstance(emb, np.ndarray):
                return emb[0].tolist()
            return emb[0]
        except Exception as e:
            self._log(f"❌ Text encoding error: {e}")
            return None

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> Optional[List[float]]:
        try:
            # AMP: GPU이고 half/bfloat이면 자동 캐스트
            use_amp = (self.device.type == "cuda") and (
                self.dtype in (torch.float16, torch.bfloat16)
            )
            if use_amp:
                amp_dtype = (
                    torch.float16 if self.dtype == torch.float16 else torch.bfloat16
                )
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    emb = self.model.encode_image([image])
            else:
                emb = self.model.encode_image([image])

            if isinstance(emb, torch.Tensor):
                return emb[0].detach().cpu().numpy().tolist()
            if isinstance(emb, np.ndarray):
                return emb[0].tolist()
            return emb[0]
        except Exception as e:
            self._log(f"❌ Image encoding error: {e}")
            return None
