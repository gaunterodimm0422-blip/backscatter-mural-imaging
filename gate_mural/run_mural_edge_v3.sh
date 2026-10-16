#!/usr/bin/env bash
# M3：新探测头构型下重跑 M2 的边界定位（几何 v2）。
#
# 改了什么（相对 M1/M2 全部运行使用的原构型）
# ------------------------------------------
#   发射锥半角   45°      -> 35°
#   D1 厚度      15 mm    -> 10 mm
#   D1 内半径    20 mm    -> 12.5 mm
#   D1 外半径    200 mm   -> 100 mm
#   D1 中面 z    17.5 mm  -> 7.5 mm（下表面离物体 2.5 mm）
#   D2 下表面 z  50 mm    -> 42.5 mm（中面 52.5，源到中面 35.0）
#   体模、源位置、能量窗、符合窗、判选链条、重建算法 —— 全部不动。
#
# 四个数为什么是自洽的一组
# ------------------------
#   1) D1 下表面 z = 2.5 时，35° 锥在该处半径 = 15.0 * tan35° = 10.50 mm，
#      内半径 12.5 mm 留 2.00 mm 余量 —— 源不会直照 D1 内壁。
#      **内半径 12.5 只有在锥收到 35° 且 D1 压到贴表面时才成立，两者绑定。**
#   2) 源到 D2 中面 35.0 mm，D2 半宽 25 mm，边缘接受半角 atan(25/35) = 35.5°，
#      与 35° 发射锥基本重合 —— 锥正好填满 D2 的内切圆。
#
# 一个会误导判据的记账陷阱（必读）
# --------------------------------
#   SOURCE_THETA_MAX_DEG 是**抽样设置**，不是硬件。真实 Na-22 源各向同性发射，
#   把锥从 45° 收到 35°，等于"同样的初级粒子数代表更多的真实湮没"。
#   立体角 2π(1-cosθ)：45° 是 1.840 sr，35° 是 1.136 sr，比值 0.617。
#   蒙卡预言：
#     每个初级粒子的选出率      新/旧 = 1.047   <- GATE 输出里直接看到的就是这个
#     每次真实湮没的选出率      新/旧 = 0.646   <- 真实测量时间要按这个算
#   **所以看到"每百万初级粒子选出的事例数几乎没变"不等于计数率没变，
#   真实计数率其实降到 0.65 倍。判据里两个都要记。**
#
# 蒙卡预言（m3v2_predict.json，见到 GATE 数据之前冻结）
# ----------------------------------------------------
#   深 30 mm 处视场半宽   27.9 mm（旧，被 D2 边缘限制） -> 33.3 mm（新，被锥限制），+19%
#   各重建深度带的空鼓侧/完好侧对比度（20 mm 厚空鼓）：
#       深度带      旧        新
#       0-8      9.5%     23.5%（该带单独拟合一向不可靠，不参与判据）
#       8-15    16.9%     13.3%
#       15-25   31.6%     37.5%
#       25-35   34.1%     44.1%
#   横向刻度系数 k 的几何预言：旧 0.819（实测 0.797±0.028，差 1.2σ）-> 新 0.844
#
# 阶段
# ----
#   STAGE=a  halfvoid（边界在轴心）+ ref（底片）+ refnull（零假设底片），三档，判据 1~8
#            refnull 与 ref 的空鼓摆位逐字相同（都挪到 x=200 看不见），只换随机种子，
#            所以 refnull/ref 是同一个物理场的两次独立实现，T 必须小 —— 这就是负对照。
#            **不能沿用 M2 的 ref_null（mural_ref_t10_L-15_2m_seed24091012）**：
#            那一档是旧几何跑的，拿它当新几何的底片，比值里会混进几何差异，
#            负对照会假报警。这是本轮相对 M2 必须多跑的一档。
#   STAGE=b  offset（边界离轴 -8 mm）+ offset2（-16 mm），两档，判据 9（重新标定 k）
#   每档 400 万事例。a 跑完先判，不通过就别跑 b。
#
# 用法
#   cd /home/vgate/beisanshe/gate_bundle_20260910/gate
#   nohup env STAGE=a SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_edge_v3.sh > edge_v3a.out 2>&1 &
#   echo "PID: $!"
#   tail -f edge_v3a.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_PRIMARIES="${N_PRIMARIES:-4000000}"
if [ -z "${SHARE_OUT:-}" ]; then
  for cand in /media/sf_vgate_shared /media/sf_share /media/sf_vgate_share; do
    if [ -d "$cand" ]; then SHARE_OUT="${cand}/output"; break; fi
  done
fi
if [ -z "${SHARE_OUT:-}" ]; then
  echo "找不到共享盘挂载点。请用 SHARE_OUT=/media/sf_xxx/output 指定。"
  ls -d /media/sf_* 2>/dev/null || true; exit 1
fi

# ---- 探测头几何 v2 ----
SOURCE_THETA_MAX_DEG=35
D1_RMIN_MM=12.5
D1_RMAX_MM=100
D1_HEIGHT_MM=10
D1_CENTER_Z_MM=7.5                 # D1 全局中面 z
D2_BOTTOM_Z_MM=42.5                # D2 下表面 z
D2_THICKNESS_MM=20
SPECTHEAD_CENTER_Z_MM=40           # 几何宏里写死的，换算用
D1_TRANSLATION_Z_MM=$(python3 -c "print(${D1_CENTER_Z_MM}-${SPECTHEAD_CENTER_Z_MM})")
D2_TRANSLATION_Z_MM=$(python3 -c "print(${D2_BOTTOM_Z_MM}+${D2_THICKNESS_MM}/2-${SPECTHEAD_CENTER_Z_MM})")

SOURCE_ACTIVITY_BQ=1000000
PHANTOM_MAC="phantom_mural.mac"
PHANTOM_VOLUME="ground_coarse"
TRUTH_MAC="truth_mural.mac"
TRUTH_SUFFIX="_truth_mural"
MAIN_MAC="main_baseline_pose_v2.mac"

# ---- 体模：与 M2 逐字相同 ----
GROUND_FINE_MM=5
GROUND_COARSE_MM=35
SUPPORT_MM=100
GROUND_MATERIAL="MuralGround"
SUPPORT_MATERIAL="Conglomerate"
VOID_MATERIAL="Air"
VOID_THICKNESS_MM=20
VOID_DEPTH_MM=20
HEAD_X_MM=-15

STAGE="${STAGE:-a}"
case "$STAGE" in
  a) EDGE_LIST="halfvoid:250:110:400:24091301 ref:120:200:120:24091302 refnull:120:200:120:24091303" ;;
  b) EDGE_LIST="halfvoidx23:250:102:400:24091311 halfvoidx31:258:98:400:24091312" ;;
  *) echo "STAGE 只能是 a 或 b"; exit 1 ;;
esac
N_JOBS=$(echo $EDGE_LIST | wc -w)

GROUND_FINE_CENTER_Z_MM=$(python3 -c "print(-${GROUND_FINE_MM}/2)")
GROUND_COARSE_CENTER_Z_MM=$(python3 -c "print(-(${GROUND_FINE_MM}+${GROUND_COARSE_MM}/2))")
SUPPORT_CENTER_Z_MM=$(python3 -c "print(-(${GROUND_FINE_MM}+${GROUND_COARSE_MM}+${SUPPORT_MM}/2))")
VOID_LOCAL_Z_MM=$(python3 -c "print(${GROUND_FINE_MM}+${GROUND_COARSE_MM}/2-${VOID_DEPTH_MM}-${VOID_THICKNESS_MM}/2)")

python3 - <<PYCHK
import math
f,c,d,t = ${GROUND_FINE_MM}, ${GROUND_COARSE_MM}, ${VOID_DEPTH_MM}, ${VOID_THICKNESS_MM}
assert d >= f, f"空鼓上表面 {d} mm 比细泥层底 {f} mm 还浅"
assert d + t <= f + c, f"空鼓下表面 {d+t} mm 超过粗泥层底 {f+c} mm"
src_z, theta = 17.5, ${SOURCE_THETA_MAX_DEG}
d1_bottom = ${D1_CENTER_Z_MM} - ${D1_HEIGHT_MM}/2
rmin_limit = (src_z - d1_bottom) * math.tan(math.radians(theta))
assert d1_bottom > 0, f"D1 下表面 z={d1_bottom} 已经插进物体里"
assert ${D1_RMIN_MM} > rmin_limit, (
    f"内半径 {${D1_RMIN_MM}} mm 小于源直照下限 {rmin_limit:.2f} mm："
    f"{theta}° 锥会打在 D1 内壁上")
d2_mid = ${D2_BOTTOM_Z_MM} + ${D2_THICKNESS_MM}/2
accept = math.degrees(math.atan(25.0/(d2_mid - src_z)))
print(f"几何自检通过")
print(f"  空鼓 z ∈ [{-(d+t)}, {-d}] mm，粗泥层 z ∈ [{-(f+c)}, {-f}] mm")
print(f"  D1 占 z ∈ [{d1_bottom}, {${D1_CENTER_Z_MM}+${D1_HEIGHT_MM}/2}] mm，"
      f"下表面离物体 {d1_bottom} mm，内半径下限 {rmin_limit:.2f} mm，"
      f"实取 {${D1_RMIN_MM}} mm，余量 {${D1_RMIN_MM}-rmin_limit:+.2f} mm")
print(f"  D2 占 z ∈ [{${D2_BOTTOM_Z_MM}}, {d2_mid + ${D2_THICKNESS_MM}/2}] mm，中面 {d2_mid} mm，"
      f"源到中面 {d2_mid-src_z} mm，边缘接受半角 {accept:.1f}°，发射锥 {theta}°")
print(f"  深 30 mm 处视场半宽 {(src_z+30)*math.tan(math.radians(min(theta, accept))):.1f} mm")
PYCHK

echo "=================================================================="
echo "M3：新探测头构型下的空鼓横向边界定位"
echo "几何      : ${MAIN_MAC} + geometry_detectors_pose_v2.mac"
echo "发射锥    : ${SOURCE_THETA_MAX_DEG}°（旧 45°）"
echo "D1        : 内半径 ${D1_RMIN_MM} / 外半径 ${D1_RMAX_MM} / 厚 ${D1_HEIGHT_MM} / 中面 z ${D1_CENTER_Z_MM} mm"
echo "D2        : 下表面 z ${D2_BOTTOM_Z_MM} / 厚 ${D2_THICKNESS_MM} mm"
echo "空鼓      : ${VOID_THICKNESS_MM} mm 厚，深 ${VOID_DEPTH_MM} mm；探测头 L = ${HEAD_X_MM} mm"
echo "阶段      : STAGE=${STAGE}   ${N_JOBS} 档，每档 ${N_PRIMARIES} 事例"
echo "共享盘输出: ${SHARE_OUT}"
echo "开始时间  : $(date '+%F %T')"
echo "=================================================================="
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1：几何自检与参数抬头已打印，不启动 GATE，退出。"
  exit 0
fi

mkdir -p "$SHARE_OUT"
JOB_N=0
for entry in $EDGE_LIST; do
  JOB_N=$((JOB_N+1))
  IFS=':' read -r KIND VOID_XLEN_MM VOID_X_MM VOID_YLEN_MM RANDOM_SEED <<< "$entry"
  VOID_Y_MM=0
  RUN_ID="m3_${KIND}_t${VOID_THICKNESS_MM}_L${HEAD_X_MM}_$((N_PRIMARIES/1000000))m_seed${RANDOM_SEED}"
  OUT_DIR="output/${RUN_ID}"
  OUTPUT_PREFIX="${OUT_DIR}/hits"

  if [ -d "${SHARE_OUT}/${RUN_ID}" ]; then
    echo "[跳过 ${JOB_N}/${N_JOBS}] 共享盘已存在 ${RUN_ID}"; echo; continue
  fi
  mkdir -p "$OUT_DIR"
  echo "=== [${JOB_N}/${N_JOBS}] ${KIND}  空鼓 x ∈ [$(python3 -c "print(${VOID_X_MM}-${VOID_XLEN_MM}/2)"), $(python3 -c "print(${VOID_X_MM}+${VOID_XLEN_MM}/2)")] mm  种子 ${RANDOM_SEED}  $(date '+%T') ==="
  df -h . | tail -1

  Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
  sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"
  {
    echo "RUN_ID=${RUN_ID}"
    echo "PURPOSE=M3 mural void lateral edge localization with head geometry v2"
    echo "KIND=${KIND}"
    echo "MAIN_MAC=${MAIN_MAC}"
    echo "SOURCE_THETA_MAX_DEG=${SOURCE_THETA_MAX_DEG}"
    echo "D1_RMIN_MM=${D1_RMIN_MM}"
    echo "D1_RMAX_MM=${D1_RMAX_MM}"
    echo "D1_HEIGHT_MM=${D1_HEIGHT_MM}"
    echo "D1_CENTER_Z_MM=${D1_CENTER_Z_MM}"
    echo "D1_TRANSLATION_Z_MM=${D1_TRANSLATION_Z_MM}"
    echo "D2_BOTTOM_Z_MM=${D2_BOTTOM_Z_MM}"
    echo "D2_TRANSLATION_Z_MM=${D2_TRANSLATION_Z_MM}"
    echo "HEAD_X_MM=${HEAD_X_MM}"
    echo "N_PRIMARIES=${N_PRIMARIES}"
    echo "RANDOM_SEED=${RANDOM_SEED}"
    echo "VOID_THICKNESS_MM=${VOID_THICKNESS_MM}"
    echo "VOID_DEPTH_MM=${VOID_DEPTH_MM}"
    echo "VOID_XLEN_MM=${VOID_XLEN_MM}"
    echo "VOID_YLEN_MM=${VOID_YLEN_MM}"
    echo "VOID_X_MM=${VOID_X_MM}"
    echo "VOID_LOCAL_Z_MM=${VOID_LOCAL_Z_MM}"
    echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
  } > "${OUT_DIR}/run_parameters.txt"

  Gate -a \
    "[OUTPUT_PREFIX,${OUTPUT_PREFIX}][N_PRIMARIES,${N_PRIMARIES}][SOURCE_ACTIVITY_BQ,${SOURCE_ACTIVITY_BQ}][RANDOM_SEED,${RANDOM_SEED}][D1_RMAX_MM,${D1_RMAX_MM}][D1_RMIN_MM,${D1_RMIN_MM}][D1_HEIGHT_MM,${D1_HEIGHT_MM}][D1_TRANSLATION_Z_MM,${D1_TRANSLATION_Z_MM}][D2_TRANSLATION_Z_MM,${D2_TRANSLATION_Z_MM}][SOURCE_THETA_MAX_DEG,${SOURCE_THETA_MAX_DEG}][HEAD_X_MM,${HEAD_X_MM}][PHANTOM_MAC,${PHANTOM_MAC}][PHANTOM_VOLUME,${PHANTOM_VOLUME}][TRUTH_MAC,${TRUTH_MAC}][TARGET_DEPTH_MM,30][TARGET_THICKNESS_MM,1][TARGET_LATERAL_MM,300][TARGET_RADIUS_MM,1][TARGET_SEPARATION_MM,10][TARGET_MATERIAL,Concrete][GROUND_FINE_MM,${GROUND_FINE_MM}][GROUND_COARSE_MM,${GROUND_COARSE_MM}][SUPPORT_MM,${SUPPORT_MM}][GROUND_MATERIAL,${GROUND_MATERIAL}][SUPPORT_MATERIAL,${SUPPORT_MATERIAL}][VOID_MATERIAL,${VOID_MATERIAL}][GROUND_FINE_CENTER_Z_MM,${GROUND_FINE_CENTER_Z_MM}][GROUND_COARSE_CENTER_Z_MM,${GROUND_COARSE_CENTER_Z_MM}][SUPPORT_CENTER_Z_MM,${SUPPORT_CENTER_Z_MM}][VOID_THICKNESS_MM,${VOID_THICKNESS_MM}][VOID_XLEN_MM,${VOID_XLEN_MM}][VOID_YLEN_MM,${VOID_YLEN_MM}][VOID_X_MM,${VOID_X_MM}][VOID_Y_MM,${VOID_Y_MM}][VOID_LOCAL_Z_MM,${VOID_LOCAL_Z_MM}]" \
    "${MAIN_MAC}" > "${OUT_DIR}/gate.log" 2>&1

  test -f "${OUTPUT_PREFIX}.root"
  test -f "${OUTPUT_PREFIX}${TRUTH_SUFFIX}.root"
  if grep -qiE "overlap|protrud" "${OUT_DIR}/gate.log"; then
    echo "  !! 日志出现 overlap/protrude 字样，请把 gate.log 发回检查"
  fi
  echo "  GATE 完成 $(date '+%T')  用时 $(grep -oE 'Real=[0-9.]+s' "${OUT_DIR}/gate.log" | tail -1)"
  echo "  hits.root = $(stat -c %s "${OUTPUT_PREFIX}.root") 字节"
  echo "  拷到共享盘 ..."
  mkdir -p "${SHARE_OUT}/${RUN_ID}"
  cp -f "${OUT_DIR}"/* "${SHARE_OUT}/${RUN_ID}/"
  test -s "${SHARE_OUT}/${RUN_ID}/hits.root"
  test -s "${SHARE_OUT}/${RUN_ID}/hits${TRUTH_SUFFIX}.root"
  if [ "${KEEP_LOCAL:-0}" = "0" ]; then
    rm -rf "${OUT_DIR}"
    echo "  本地副本已删除，根分区剩余 $(df -h . | tail -1 | awk '{print $4}')"
  fi
  echo "  完成 $(date '+%T')"; echo
done

echo "=================================================================="
echo "全部结束 $(date '+%F %T')"
echo "=================================================================="
