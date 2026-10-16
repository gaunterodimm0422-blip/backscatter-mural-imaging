#!/usr/bin/env bash
# S2V：混凝土体内空洞的多位姿深度定位。
#
# 目的
# ----
# S2 证明了多视角能把深度 FWHM 从 38.5 mm 压到 3.86 mm，但那是"空气里的一根
# 混凝土细杆"——一个孤立的亮目标。工程需求是"混凝土里的空洞"——亮背景上的
# 暗缺损。似然相乘这个估计器是为定位亮点设计的，对负信号是否还成立没有验过。
# 这一轮就是验它。
#
# 体模
# ----
# phantom_concrete_void.mac：500x500x100 mm 混凝土板（顶面 z=0），
# 沿 y 的空洞圆柱，**直径 20 mm**，深 30 mm（与 S2 细杆同深度）。
#
# 空洞尺寸：先用 20 mm。合成体模上标定估计器时量到，10 mm 空洞在 200 万事件下
# 检出统计量只有噪声的 1.33 倍——太勉强，做不出可信结论；20 mm 是 2.87 倍。
# 20 mm 的蜂窝/空鼓在工程上本来就是典型缺陷。"能查到多小"是后续单独的题。
#
# 位姿：L = -15 mm 与 +15 mm，对目标张开各约 17.5 度，合计 35 度。
#   为什么不是 S2 用的 ±27：深 30 mm 处的照射半宽只有 ±27.9 mm
#   (D2 半宽 25 x 47.5/42.5)。L=27 会把空洞推到照射锥边缘，约 40% 的空洞
#   根本照不到，测出来的"阴影"会和孔径边缘混在一起。L=15 时空洞完全在
#   照射区内。代价是张开角小，解析预测深度 8.3 mm（±27 时是 5.1 mm）。
#
# 参照跑
# ------
# 每个位姿都配一个"无空洞"参照：**同一个体模宏**，只把 VOID_X_MM 从 0 挪到
# 200 mm（挪出视野，仍在板内）。几何、材料、truth actor 完全一致，只有空洞
# 位置不同——这是能做到的最干净的对照。有了它，孔径衰减在相除时精确抵消，
# 而且可以用"参照对参照"做负对照，量出零假设下的假峰有多大。
#
# 四档 x 200 万事件。每档 ROOT 约 1.2 GB（归档的 2M 空洞跑就是 1.16 GB）。
# 耗时：空气细杆 5M 用了 2h42m；混凝土整板每个初级粒子的输运量大得多，
# 我估计每档 2-3 小时，四档合计 8-12 小时，**这个估计不确定度较大**。
# 第一档跑完日志里会打出实际耗时，届时按比例外推即可。
#
# 用法（务必 nohup，别放前台）：
#     cd /home/vgate/beisanshe_v1/gate
#     nohup bash run_s2_void.sh > s2v.out 2>&1 &
#     echo "PID: $!"
# 看进度：
#     tail -f /home/vgate/beisanshe_v1/gate/s2v.out
# 中断后重跑：已经拷到共享盘的档会自动跳过。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_PRIMARIES="${N_PRIMARIES:-2000000}"
SHARE_OUT="${SHARE_OUT:-/media/sf_share/beisanshe_v1/gate/output}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"

SOURCE_ACTIVITY_BQ=1000000
D1_RMAX_MM=200
SOURCE_THETA_MAX_DEG=45
PHANTOM_MAC="phantom_concrete_void.mac"
PHANTOM_VOLUME="concrete"
TRUTH_MAC="truth_concrete.mac"
TRUTH_SUFFIX="_truth_concrete"
VOID_RADIUS_MM=10          # 半径 10 mm -> 直径 20 mm（见脚本头"空洞尺寸"一节）
VOID_LENGTH_MM=300
VOID_DEPTH_MM=30
VOID_LOCAL_Z_MM=20         # = 50 - VOID_DEPTH_MM
# 细杆体模的参数在本轮用不到，但宏里的占位符必须都给值
TARGET_DEPTH_MM=30
TARGET_RADIUS_MM=1
TARGET_LATERAL_MM=300
TARGET_MATERIAL="Concrete"
TARGET_THICKNESS_MM=1
TARGET_SEPARATION_MM=10

# 顺序：先把 L=-15 这一对（空洞+参照）跑完，中途停掉也能先做单位姿分析
# 每档: 标签:位姿:空洞x:种子
JOBS="void:-15:0:24090111 ref:-15:200:24090112 void:15:0:24090113 ref:15:200:24090114"

echo "=================================================================="
echo "S2V：混凝土体内空洞，多位姿深度定位"
echo "体模      : ${PHANTOM_MAC}（直径 $((VOID_RADIUS_MM*2)) mm 空洞，深 ${VOID_DEPTH_MM} mm）"
echo "位姿      : L = -15 mm 与 +15 mm，各配一个 VOID_X_MM=200 的无空洞参照"
echo "每档事件  : ${N_PRIMARIES}"
echo "共享盘输出: ${SHARE_OUT}"
echo "开始时间  : $(date '+%F %T')"
echo "=================================================================="
echo

mkdir -p "$SHARE_OUT"
JOB_N=0
for entry in $JOBS; do
  JOB_N=$((JOB_N+1))
  IFS=':' read -r KIND HEAD_X_MM VOID_X_MM RANDOM_SEED <<< "$entry"
  RUN_ID="s2v_${KIND}_L${HEAD_X_MM}_$((N_PRIMARIES/1000000))m_seed${RANDOM_SEED}"
  OUT_DIR="output/${RUN_ID}"
  OUTPUT_PREFIX="${OUT_DIR}/hits"

  if [ -d "${SHARE_OUT}/${RUN_ID}" ]; then
    echo "[跳过 ${JOB_N}/4] 共享盘已存在 ${RUN_ID}"
    echo
    continue
  fi

  mkdir -p "$OUT_DIR"
  echo "=== [${JOB_N}/4] ${KIND}  L = ${HEAD_X_MM} mm  空洞x = ${VOID_X_MM} mm  种子 ${RANDOM_SEED}   $(date '+%T') ==="
  df -h /home/vgate | tail -1

  Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
  sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"
  {
    echo "RUN_ID=${RUN_ID}"
    echo "PURPOSE=S2V multiview void-in-concrete depth localization"
    echo "KIND=${KIND}"
    echo "HEAD_X_MM=${HEAD_X_MM}"
    echo "PHANTOM_MAC=${PHANTOM_MAC}"
    echo "PHANTOM_VOLUME=${PHANTOM_VOLUME}"
    echo "TRUTH_MAC=${TRUTH_MAC}"
    echo "N_PRIMARIES=${N_PRIMARIES}"
    echo "RANDOM_SEED=${RANDOM_SEED}"
    echo "D1_RMAX_MM=${D1_RMAX_MM}"
    echo "SOURCE_THETA_MAX_DEG=${SOURCE_THETA_MAX_DEG}"
    echo "VOID_RADIUS_MM=${VOID_RADIUS_MM}"
    echo "VOID_LENGTH_MM=${VOID_LENGTH_MM}"
    echo "VOID_X_MM=${VOID_X_MM}"
    echo "VOID_DEPTH_MM=${VOID_DEPTH_MM}"
    echo "VOID_LOCAL_Z_MM=${VOID_LOCAL_Z_MM}"
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
  echo "  GATE 完成 $(date '+%T')  用时 $(grep -oE 'Real=[0-9.]+s' "${OUT_DIR}/gate.log" | tail -1)"
  echo "  hits.root = $(stat -c %s "${OUTPUT_PREFIX}.root") 字节"

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
echo "S2V 四档全部完成  $(date '+%F %T')"
ls -d "${SHARE_OUT}"/s2v_* 2>/dev/null || true
echo "=================================================================="
