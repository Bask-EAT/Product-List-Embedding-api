# utils_vector.py (bfloat16/float16/torch 텐서 안전 지원)
from __future__ import annotations
from typing import Iterable, Sequence, List, Any
import numpy as np

__all__ = ["l2_normalize", "dot_score", "cosine_score"]


def _to_f32(x: Any) -> np.ndarray:
    """
    임베딩을 float32 ndarray로 변환.
    - list/tuple/ndarray/torch.Tensor(bfloat16/float16 포함) 허용
    - NaN/Inf -> 0 처리
    """
    # torch 텐서 지원(선택)
    if "torch" in globals() or "torch" in str(type(x)):
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.detach().to(dtype=torch.float32).cpu().numpy()
        except Exception:
            pass

    a = np.asarray(x)
    # bfloat16 안전 캐스팅 (NumPy가 지원하면 dtype 이름이 'bfloat16')
    try:
        if a.dtype.name == "bfloat16":
            a = a.astype(np.float32, copy=False)
    except Exception:
        pass

    # 그 외는 모두 float32로
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)

    # NaN/Inf 방어
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def l2_normalize(vec: Sequence[float]) -> List[float]:
    v = _to_f32(vec)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


def dot_score(a: Sequence[float], b: Sequence[float]) -> float:
    A = _to_f32(a)
    B = _to_f32(b)
    m = min(A.shape[0], B.shape[0])
    if m == 0:
        return 0.0
    return float(np.dot(A[:m], B[:m]))
