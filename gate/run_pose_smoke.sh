#!/usr/bin/env bash
# S0b 冒烟测试：验证"探测头横向平移"这个位姿实现是否正确。
#
# 体模默认用 100 mm 完整水泥板（信号量足够）。第一版用点体模是错的：
# 点体模在真空里、2 万事件下靶信号期望值不到 1 个，测到的全是 D2->D1 串扰。
#
# 探测头（源 + D1 + D2）依次平移到 L = 0 / 15 / 27 / 40 mm，物体不动。
# 水泥板横向 500 mm、均匀，所以在探测头自身坐标系里，四档的分布应当完全重合。
# 任何不重合都说明平移实现有问题。
#
# 用法（在 gate 目录下）：
#     bash run_pose_smoke.sh
# 可选：
#     N_PRIMARIES=200000 bash run_pose_smoke.sh
#     PHANTOM_KIND=point bash run_pose_smoke.sh      # 保留点体模选项
#     HEAD_X_LIST="0 20" bash run_pose_smoke.sh
#
# 10 万事件 x 4 档，预计十几分钟。通过后再跑 S1/S2 的正式任务。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_PRIMARIES="${N_PRIMARIES:-100000}"
SOURCE_ACTIVITY_BQ="${SOURCE_ACTIVITY_BQ:-1000000}"
RANDOM_SEED="${RANDOM_SEED:-24083001}"
D1_RMAX_MM="${D1_RMAX_MM:-200}"
SOURCE_THETA_MAX_DEG="${SOURCE_THETA_MAX_DEG:-45}"
HEAD_X_LIST="${HEAD_X_LIST:-0 15 27 40}"
PHANTOM_KIND="${PHANTOM_KIND:-concrete}"
RUN_TAG="${RUN_TAG:-pose_smoke}"

case "$PHANTOM_KIND" in
  concrete)
    PHANTOM_MAC="phantom_concrete.mac"
    PHANTOM_VOLUME="concrete"
    TRUTH_MAC="truth_concrete.mac"
    TRUTH_OUTPUT_SUFFIX="_truth_concrete"
    ;;
  point)
    PHANTOM_MAC="phantom_point.mac"
    PHANTOM_VOLUME="target"
    TRUTH_MAC="truth_target.mac"
    TRUTH_OUTPUT_SUFFIX="_truth_target"
    ;;
  *)
    echo "PHANTOM_KIND 只支持 concrete 或 point，收到：${PHANTOM_KIND}" >&2
    exit 2
    ;;
esac

# 下面这些别名 phantom_concrete.mac 用不到，但 Gate 要求命令行里出现的
# 别名集合覆盖所有被载入宏的引用，保留默认值最安全。
TARGET_DEPTH_MM="${TARGET_DEPTH_MM:-30}"
TARGET_RADIUS_MM="${TARGET_RADIUS_MM:-1}"
TARGET_THICKNESS_MM="${TARGET_THICKNESS_MM:-1}"
TARGET_MATERIAL="${TARGET_MATERIAL:-Concrete}"
TARGET_LATERAL_MM=300
TARGET_SEPARATION_MM=10
VOID_RADIUS_MM=5
VOID_LENGTH_MM=300
VOID_X_MM=0
VOID_LOCAL_Z_MM=20

echo "体模：${PHANTOM_MAC}    每档初级事件：${N_PRIMARIES}    位姿：${HEAD_X_LIST}"
echo

for HEAD_X_MM in ${HEAD_X_LIST}; do
  RUN_ID="${RUN_TAG}_L${HEAD_X_MM}"
  OUT_DIR="output/${RUN_ID}"
  OUTPUT_PREFIX="${OUT_DIR}/hits"
  mkdir -p "$OUT_DIR"

  Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
  sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"

  {
    echo "RUN_ID=${RUN_ID}"
    echo "HEAD_X_MM=${HEAD_X_MM}"
    echo "PHANTOM_KIND=${PHANTOM_KIND}"
    echo "PHANTOM_MAC=${PHANTOM_MAC}"
    echo "PHANTOM_VOLUME=${PHANTOM_VOLUME}"
    echo "TRUTH_MAC=${TRUTH_MAC}"
    echo "N_PRIMARIES=${N_PRIMARIES}"
    echo "RANDOM_SEED=${RANDOM_SEED}"
    echo "D1_RMAX_MM=${D1_RMAX_MM}"
    echo "SOURCE_THETA_MAX_DEG=${SOURCE_THETA_MAX_DEG}"
    echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
  } > "${OUT_DIR}/run_parameters.txt"

  echo "=== 位姿 L = ${HEAD_X_MM} mm ==="
  Gate -a \
    "[OUTPUT_PREFIX,${OUTPUT_PREFIX}][N_PRIMARIES,${N_PRIMARIES}][SOURCE_ACTIVITY_BQ,${SOURCE_ACTIVITY_BQ}][RANDOM_SEED,${RANDOM_SEED}][D1_RMAX_MM,${D1_RMAX_MM}][SOURCE_THETA_MAX_DEG,${SOURCE_THETA_MAX_DEG}][HEAD_X_MM,${HEAD_X_MM}][PHANTOM_MAC,${PHANTOM_MAC}][PHANTOM_VOLUME,${PHANTOM_VOLUME}][TRUTH_MAC,${TRUTH_MAC}][TARGET_DEPTH_MM,${TARGET_DEPTH_MM}][TARGET_THICKNESS_MM,${TARGET_THICKNESS_MM}][TARGET_LATERAL_MM,${TARGET_LATERAL_MM}][TARGET_RADIUS_MM,${TARGET_RADIUS_MM}][TARGET_SEPARATION_MM,${TARGET_SEPARATION_MM}][TARGET_MATERIAL,${TARGET_MATERIAL}][VOID_RADIUS_MM,${VOID_RADIUS_MM}][VOID_LENGTH_MM,${VOID_LENGTH_MM}][VOID_X_MM,${VOID_X_MM}][VOID_LOCAL_Z_MM,${VOID_LOCAL_Z_MM}]" \
    main_baseline_pose.mac > "${OUT_DIR}/gate.log" 2>&1

  test -f "${OUTPUT_PREFIX}.root"
  test -f "${OUTPUT_PREFIX}${TRUTH_OUTPUT_SUFFIX}.root"
  if grep -qi "overlap" "${OUT_DIR}/gate.log"; then
    echo "  !! 日志里出现 overlap 字样，几何可能有重叠，请把 gate.log 发回检查"
  fi
  echo "  完成 -> ${OUT_DIR}"
done

echo
echo "全部完成。把 output/${RUN_TAG}_L* 回传到 Windows 的"
echo "  C:\\Users\\HOUWENDI\\Documents\\beisanshe\\runs\\"
echo "然后运行："
echo "  E:\\python3.12\\python.exe C:\\Users\\HOUWENDI\\Documents\\beisanshe\\scripts\\check_pose_smoke.py"
