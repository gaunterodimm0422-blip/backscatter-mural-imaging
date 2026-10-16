"""位姿感知的事件筛选（新增文件，不改动已冻结的 pipeline.build_selection）。

背景
----
``pipeline.build_selection`` 里有两条几何判据是相对**世界原点**写的：

    d1_radius = hypot(d1_x_mm, d1_y_mm)                    # D1 环形有效半径
    |d2_x_mm| <= half  且  |d2_y_mm| <= half               # D2 有效面积

探测头不平移（L = 0）时世界原点就是探测头轴心，两条判据正确。一旦探测头沿 x
平移 L mm（S2 多位姿），轴心移到 x = L，这两条判据就在错误的中心上切割。
L = -27 mm 的实测后果：d2_active_area 一刀把 154 个事件砍成 0。

做法
----
不重写判据，而是把坐标换到**探测头坐标系**之后再喂给原来的 build_selection：
源与 D1、D2 是同一个刚体（SPECThead）一起平移的，源的 (x, y) 就是探测头轴心，
所以减掉逐事件的源横向坐标即可。入射极角与康普顿运动学都是平移不变量，因此
除那两条几何判据外，结果与原实现逐位相同；L = 0 时平移量为 0，是恒等变换，
历史结果不受任何影响。

返回的掩码对应**原始（世界坐标）**的 pairs 行，可直接交给重建使用。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from .pipeline import build_selection

__all__ = ["to_head_frame", "head_offsets_mm", "build_selection_pose_aware"]

# 随探测头一起平移的横向坐标；z 方向不平移，故不在此列。
_X_KEYS = ("d1_x_mm", "d2_x_mm", "source_x_mm")
_Y_KEYS = ("d1_y_mm", "d2_y_mm", "source_y_mm")


def head_offsets_mm(pairs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """逐事件的探测头轴心横向坐标 (x, y)，等于源的横向坐标。"""
    return (
        np.asarray(pairs["source_x_mm"], dtype=np.float64),
        np.asarray(pairs["source_y_mm"], dtype=np.float64),
    )


def to_head_frame(pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """返回一份浅拷贝，其中横向坐标已换算到探测头坐标系（轴心为原点）。"""
    hx, hy = head_offsets_mm(pairs)
    shifted = dict(pairs)
    for key in _X_KEYS:
        shifted[key] = np.asarray(pairs[key], dtype=np.float64) - hx
    for key in _Y_KEYS:
        shifted[key] = np.asarray(pairs[key], dtype=np.float64) - hy
    return shifted


def build_selection_pose_aware(
    pairs: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[np.ndarray, "OrderedDict[str, int]", dict[str, Any]]:
    """与 pipeline.build_selection 同签名，但 D1 半径 / D2 有效面积以探测头为中心。"""
    selected, chain, diagnostics = build_selection(to_head_frame(pairs), config)
    hx, hy = head_offsets_mm(pairs)
    diagnostics = dict(diagnostics)
    diagnostics["head_offset_x_mm"] = np.unique(np.round(hx, 6)).tolist()
    diagnostics["head_offset_y_mm"] = np.unique(np.round(hy, 6)).tolist()
    diagnostics["pose_aware_selection"] = True
    return selected, chain, diagnostics
