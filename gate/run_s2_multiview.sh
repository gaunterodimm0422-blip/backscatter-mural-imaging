#!/usr/bin/env bash
# S2（窄版）：多位姿深度分辨率验证。
#
# 目的：验证导师提出的"移动探测头、多视角联合重建"能否把深度 FWHM
#       从单视角的 38.5 mm 降到毫米级。解析估算：两视角张开 ±30° -> 5.1 mm。
#
# 体模：沿 y 的 2 mm 细杆，深 30 mm，位于 x=0（与官方横向 FWHM 用的是同一个体模）。
#       细杆在 x 和深度两个方向都是局域的，探测头沿 x 平移时两个位姿的
#       误差"针"在 x-深度平面里交叉，正是要检验的效应。
#
# 位姿：L = -27 mm 和 +27 mm。对目标张开角各约 29.6°，合计约 59°。
#       不跑 L=0：解析表显示加中间视角几乎无增益（5.08 vs 5.12 mm），
#       而单视角基线已有官方数据（38.47 mm）。
#
# 每档 500 万事件，约 3 小时，ROOT 约 2.6 GB。
# 跑完立刻拷到共享盘并校验，校验通过才删本地，避免虚拟机磁盘被撑满。
#
# 用法（务必用 nohup，别放前台）：
#     cd /home/vgate/beisanshe_v1/gate
#     nohup bash run_s2_multiview.sh > s2.out 2>&1 &
#     echo "PID: $!"
# 看进度：
#     tail -f /home/vgate/beisanshe_v1/gate/s2.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_PRIMARIES="${N_PRIMARIES:-5000000}"
SHARE_OUT="${SHARE_OUT:-/media/sf_share/beisanshe_v1/gate/output}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"        # 设为 1 则拷完不删本地

SOURCE_ACTIVITY_BQ=1000000
D1_RMAX_MM=200
SOURCE_THETA_MAX_DEG=45
PHANTOM_MAC="phantom_rod.mac"        # 沿 y 的细杆
PHANTOM_VOLUME="target"
TRUTH_MAC="truth_target.mac"
TRUTH_SUFFIX="_truth_target"
TARGET_DEPTH_MM=30
TARGET_RADIUS_MM=1                   # 半径 1 mm -> 直径 2 mm
TARGET_LATERAL_MM=300
TARGET_MATERIAL="Concrete"
TARGET_THICKNESS_MM=1
TARGET_SEPARATION_MM=10
VOID_RADIUS_MM=5
VOID_LENGTH_MM=300
VOID_X_MM=0
VOID_LOCAL_Z_MM=20

# 位姿 -> 随机种子（两档相互独立）
POSES="-27:24083201 27:24083202"

echo "=================================================================="
echo "S2 窄版：多位姿深度分辨率验证"
echo "体模      : ${PHANTOM_MAC}（沿 y 的 2 mm 细杆，深 ${TARGET_DEPTH_MM} mm）"
echo "位姿      : L = -27 mm 和 +27 mm"
echo "每档事件  : ${N_PRIMARIES}"
echo "共享盘输出: ${SHARE_OUT}"
echo "开始时间  : $(date '+%F %T')"
echo "=================================================================="
echo

mkdir -p "$SHARE_OUT"

for entry in $POSES; do
  HEAD_X_MM="${entry%%:*}"
  RANDOM_SEED="${entry##*:}"
  RUN_ID="s2_rod_L${HEAD_X_MM}_5m_seed${RANDOM_SEED}"
  OUT_DIR="output/${RUN_ID}"
  OUTPUT_PREFIX="${OUT_DIR}/hits"

  if [ -d "${SHARE_OUT}/${RUN_ID}" ]; then
    echo "[跳过] 共享盘已存在 ${RUN_ID}，如需重跑请先删除它"
    echo
    continue
  fi

  mkdir -p "$OUT_DIR"
  echo "=== 位姿 L = ${HEAD_X_MM} mm   种子 ${RANDOM_SEED}   $(date '+%T') ==="
  df -h /home/vgate | tail -1

  Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
  sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"
  {
    echo "RUN_ID=${RUN_ID}"
    echo "PURPOSE=S2 multiview depth resolution, narrow scope"
    echo "HEAD_X_MM=${HEAD_X_MM}"
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
    "[OUTPUT_PREFIX,${OUTPUT_PREFIX}][N_PRIMARIES,${N_PRIMARIES}][SOURCE_ACTIVITY_BQ,${SOURCE_ACTIVITY_BQ}][RANDOM_SEED,${RANDOM_SEED}][D1_RMAX_MM,${D1_RMAX_MM}][SOURCE_THETA_MAX_DEG,${SOURCE_THETA_MAX_DEG}][HEAD_X_MM,${HEAD_X_MM}][PHANTOM_MAC,${PHANTOM_MAC}][PHANTOM_VOLUME,${PHANTOM_VOLUME}][TRUTH_MAC,${TRUTH_MAC}][TARGET_DEPTH_MM,${TARGET_DEPTH_MM}][TARGET_THICKNESS_MM,${TARGET_THICKNESS_MM}][TARGET_LATERAL_MM,${TARGET_LATERAL_MM}][TARGET_RADIUS_MM,${TARGET_RADIUS_MM}][TARGET_SEPARATION_MM,${TARGET_SEPARATION_MM}][TARGET_MATERIAL,${TARGET_MATERIAL}][VOID_RADIUS_MM,${VOID_RADIUS_MM}][VOID_LENGTH_MM,${VOID_LENGTH_MM}][VOID_X_MM,${VOID_X_MM}][VOID_LOCAL_Z_MM,${VOID_LOCAL_Z_MM}]" \
    main_baseline_pose.mac > "${OUT_DIR}/gate.log" 2>&1

  test -f "${OUTPUT_PREFIX}.root"
  test -f "${OUTPUT_PREFIX}${TRUTH_SUFFIX}.root"
  if grep -qi "overlap" "${OUT_DIR}/gate.log"; then
    echo "  !! 日志出现 overlap 字样，请把 gate.log 发回检查"
  fi
  echo "  GATE 完成 $(date '+%T')，hits.root = $(stat -c %s "${OUTPUT_PREFIX}.root") 字节"

  echo "  拷到共享盘 ..."
  cp -a "$OUT_DIR" "${SHARE_OUT}/"
  local_size=$(stat -c %s "${OUTPUT_PREFIX}.root")
  share_size=$(stat -c %s "${SHARE_OUT}/${RUN_ID}/hits.root")
  if [ "$local_size" != "$share_size" ]; then
    echo "  !! 拷贝后字节数不一致（本地 ${local_size} / 共享盘 ${share_size}），保留本地副本，中止"
    exit 1
  fi
  echo "  校验通过：${share_size} 字节一致"

  if [ "$KEEP_LOCAL" = "1" ]; then
    echo "  KEEP_LOCAL=1，保留本地副本"
  else
    rm -rf "$OUT_DIR"
    echo "  已删除本地副本，释放磁盘"
  fi
  df -h /home/vgate | tail -1
  echo
done

echo "=================================================================="
echo "S2 两档全部完成  $(date '+%F %T')"
echo "共享盘结果："
ls -d "${SHARE_OUT}"/s2_rod_L*  2>/dev/null || true
echo
echo "接下来在 Windows 上对每一档依次做（把 <RUN_ID> 换成上面两个目录名）："
echo "  cd C:\\Users\\HOUWENDI\\Documents\\beisanshe"
echo "  E:\\python3.12\\python.exe -m backscatter analyze --root \"E:\\vbgate\\share\\beisanshe_v1\\gate\\output\\<RUN_ID>\\hits.root\" --truth \"E:\\vbgate\\share\\beisanshe_v1\\gate\\output\\<RUN_ID>\\hits_truth_target.root\" --config \"configs\\transverse_fwhm_5mm_frozen.json\" --out-dir \"runs\\<RUN_ID>\""
echo "  E:\\python3.12\\python.exe scripts\\apply_response_to_analysis.py --analysis-dir runs\\<RUN_ID>"
echo "=================================================================="
