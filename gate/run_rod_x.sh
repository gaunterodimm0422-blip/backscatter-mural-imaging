#!/usr/bin/env bash
# S1：沿 x 轴放置的 2 mm 细杆，用于测量 y 方向的横向分辨率。
#
# 背景：现有的 3.67 mm 横向 FWHM 是用沿 y 的细杆测 x 方向得到的；
# 重建目前把 y 方向积分掉了（y_half_width=150 mm），所以 y 方向至今没有任何实测数。
# 把细杆转 90° 沿 x 放，就能测 y。几何上 x/y 对称，预期同样是 3.67 mm 附近——
# 这是一次带预期值的检验：对上了说明流程能出二维分辨率，对不上说明重建链路有问题。
#
# 参数与官方 x 方向那次完全一致（2 mm 直径、深 30 mm、500 万事件），
# 只换随机种子，保证两个方向的数字可以直接对比。
#
# 用法（在虚拟机本地的 gate 目录下）：
#     bash run_rod_x.sh
# 可选：
#     N_PRIMARIES=1000000 bash run_rod_x.sh        # 先跑个小的看看
#     RUN_ID=my_id bash run_rod_x.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_PRIMARIES="${N_PRIMARIES:-5000000}"
RANDOM_SEED="${RANDOM_SEED:-24083101}"
RUN_ID="${RUN_ID:-rod_x_depth30_${N_PRIMARIES}_seed${RANDOM_SEED}}"

SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-1000000}"
D1_RMAX_MM="${D1_RMAX_MM:-200}"
SOURCE_THETA_MAX_DEG="${SOURCE_THETA_MAX_DEG:-45}"
PHANTOM_MAC="phantom_rod_x.mac"
PHANTOM_VOLUME="target"
TRUTH_MAC="truth_target.mac"
TRUTH_OUTPUT_SUFFIX="_truth_target"
TARGET_DEPTH_MM="${TARGET_DEPTH_MM:-30}"
TARGET_RADIUS_MM="${TARGET_RADIUS_MM:-1}"
TARGET_LATERAL_MM="${TARGET_LATERAL_MM:-300}"
TARGET_MATERIAL="${TARGET_MATERIAL:-Concrete}"
TARGET_THICKNESS_MM=1
TARGET_SEPARATION_MM=10
VOID_RADIUS_MM=5
VOID_LENGTH_MM=300
VOID_X_MM=0
VOID_LOCAL_Z_MM=20

OUT_DIR="output/${RUN_ID}"
OUTPUT_PREFIX="${OUT_DIR}/hits"
mkdir -p "$OUT_DIR"

echo "体模：${PHANTOM_MAC}（沿 x 的 $(awk -v r="${TARGET_RADIUS_MM}" 'BEGIN{printf "%.3g", 2*r}') mm 细杆，深 ${TARGET_DEPTH_MM} mm）"
echo "初级事件：${N_PRIMARIES}    随机种子：${RANDOM_SEED}"
echo "输出：${OUT_DIR}"
echo

Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"

{
  echo "RUN_ID=${RUN_ID}"
  echo "PHANTOM_MAC=${PHANTOM_MAC}"
  echo "PHANTOM_VOLUME=${PHANTOM_VOLUME}"
  echo "TRUTH_MAC=${TRUTH_MAC}"
  echo "N_PRIMARIES=${N_PRIMARIES}"
  echo "RANDOM_SEED=${RANDOM_SEED}"
  echo "D1_RMAX_MM=${D1_RMAX_MM}"
  echo "SOURCE_THETA_MAX_DEG=${SOURCE_THETA_MAX_DEG}"
  echo "TARGET_DEPTH_MM=${TARGET_DEPTH_MM}"
  echo "TARGET_RADIUS_MM=${TARGET_RADIUS_MM}"
  echo "TARGET_LATERAL_MM=${TARGET_LATERAL_MM}"
  echo "TARGET_MATERIAL=${TARGET_MATERIAL}"
  echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
} > "${OUT_DIR}/run_parameters.txt"

Gate -a \
  "[OUTPUT_PREFIX,${OUTPUT_PREFIX}][N_PRIMARIES,${N_PRIMARIES}][SOURCE_ACTIVITY_BQ,${SOURCE_ACTIVITY_BQ}][RANDOM_SEED,${RANDOM_SEED}][D1_RMAX_MM,${D1_RMAX_MM}][SOURCE_THETA_MAX_DEG,${SOURCE_THETA_MAX_DEG}][PHANTOM_MAC,${PHANTOM_MAC}][PHANTOM_VOLUME,${PHANTOM_VOLUME}][TRUTH_MAC,${TRUTH_MAC}][TARGET_DEPTH_MM,${TARGET_DEPTH_MM}][TARGET_THICKNESS_MM,${TARGET_THICKNESS_MM}][TARGET_LATERAL_MM,${TARGET_LATERAL_MM}][TARGET_RADIUS_MM,${TARGET_RADIUS_MM}][TARGET_SEPARATION_MM,${TARGET_SEPARATION_MM}][TARGET_MATERIAL,${TARGET_MATERIAL}][VOID_RADIUS_MM,${VOID_RADIUS_MM}][VOID_LENGTH_MM,${VOID_LENGTH_MM}][VOID_X_MM,${VOID_X_MM}][VOID_LOCAL_Z_MM,${VOID_LOCAL_Z_MM}]" \
  main_baseline.mac > "${OUT_DIR}/gate.log" 2>&1

test -f "${OUTPUT_PREFIX}.root"
test -f "${OUTPUT_PREFIX}${TRUTH_OUTPUT_SUFFIX}.root"
if grep -qi "overlap" "${OUT_DIR}/gate.log"; then
  echo "!! 日志里出现 overlap 字样，请把 gate.log 发回检查"
fi

echo "完成 -> ${OUT_DIR}"
echo "把这个目录拷回共享盘：cp -a ${OUT_DIR} /media/sf_share/beisanshe_v1/gate/output/"
