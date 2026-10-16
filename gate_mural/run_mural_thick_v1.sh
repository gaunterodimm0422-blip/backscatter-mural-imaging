#!/usr/bin/env bash
# F3：空鼓厚度扫描（M3 构型），2026-09-23 冻结。
#
# 由 run_mural_edge_v3.sh 复制后改出；v3 本身一个字没动（它的 sha256 仍是 M3 冻结值 abd0a4c1…）。
#
# 要回答的问题
# ------------
#   M3 的全部好结果（偏差 −0.07 mm、k = 0.926）都来自同一个情形：20 mm 厚的空气层。
#   本轮回答：空鼓变薄到多少，还能 (1) 检出、(2) 把横向边界定到 2 mm 以内。
#
# 相对 M3 STAGE=a 的 halfvoid 档，只改一件事：空鼓厚度
# ------------------------------------------------------
#   厚度          10 / 5 / 3 / 2 / 1 mm（网格出自《壁画体模定义》第六节的预定扫描）
#   放法          下表面始终贴地仗/支撑体界面（深 40 mm），上表面深 = 40 − 厚度。
#                 这与 M3 的 20 mm 档是同一种放法（它占 20–40 mm），界面脱空也是最常见的一类空鼓。
#   不变          探测头构型、源、能窗、符合窗、判选、重建、边界位置 x = −15 mm、横向尺寸
#                 x ∈ [−15, 235] × y 400 mm、每档 400 万事例。
#   参照场        **不重跑**，沿用 M3 的 m3_ref_t20_L-15_4m_seed24091302 与
#                 m3_refnull_t20_L-15_4m_seed24091303：两者的空鼓都挪在 x ∈ [140, 260]，
#                 离照射区（深 40 mm 处半宽约 40 mm）一百多毫米，厚度多少与它们无关。
#
# 判据（见到数据之前冻结，完整版在 configs/m3_thick_t*_20260923.json 与项目文档）
# ------------------------------------------------------------------------------
#   F3-1 检出  T >= 30，且深两带（15–25、25–35）c_b 至少一个 > 0
#   F3-2 定位  F3-1 且 |x_e − (−15)| <= 2.0 mm 且 68% 区间半宽 <= 2.0 mm
#   F3-3 交叉  F3-1 的档，换零假设底片后 x_e 相差 <= 5 mm
#   检出下限 = 过 F3-1 的最薄一档；定位下限 = 过 F3-1/2/3 的最薄一档。
#   分析脚本自己打印的判据 1–5 是 M3 主档的，F3 不看。
#
# 选网格用的粗估（不是判据）
# --------------------------
#   T ∝ 厚度²，以 20 mm 档 T = 632.6 为锚：10 mm ≈ 158，5 mm ≈ 40，3 mm ≈ 14，2 mm ≈ 6，1 mm ≈ 1.6。
#   本项目事例级预言已有六条失效，量级可能差 2–5 倍，所以网格把 1–10 mm 全覆盖。
#
# 时长与空间
# ----------
#   M3 实测每档 400 万约 3 小时 05 分，五档约 15–16 小时。hits.root 每档约 3.5 GB，共享盘共约 17.5 GB；
#   根分区每次只临时放一档（跑完拷走即删）。
#   顺序由厚到薄：中途停机时，先拿到的是信息量最大的几档；再次启动会自动跳过共享盘上已有的档。
#
# 用法（虚拟机终端）
#   cd /home/vgate/beisanshe/gate_bundle_20260910/gate
#   bash -n run_mural_thick_v1.sh
#   DRY_RUN=1 SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_thick_v1.sh
#   nohup env SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_thick_v1.sh > thick_v1.out 2>&1 &
#   echo "PID: $!"
#   tail -f thick_v1.out

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
INTERFACE_DEPTH_MM=$((GROUND_FINE_MM + GROUND_COARSE_MM))   # 地仗/支撑体界面深度，= 40 mm
HEAD_X_MM=-15

# 条目格式  KIND:VOID_XLEN_MM:VOID_X_MM:VOID_YLEN_MM:RANDOM_SEED:VOID_THICKNESS_MM
# 横向与 M3 STAGE=a 的 halfvoid 逐字相同（250:110:400 → x ∈ [-15, 235]），只改最后一项厚度。
# 顺序由厚到薄。
EDGE_LIST="halfvoid:250:110:400:24092301:10 halfvoid:250:110:400:24092302:5 halfvoid:250:110:400:24092303:3 halfvoid:250:110:400:24092304:2 halfvoid:250:110:400:24092305:1"
N_JOBS=$(echo $EDGE_LIST | wc -w)

GROUND_FINE_CENTER_Z_MM=$(python3 -c "print(-${GROUND_FINE_MM}/2)")
GROUND_COARSE_CENTER_Z_MM=$(python3 -c "print(-(${GROUND_FINE_MM}+${GROUND_COARSE_MM}/2))")
SUPPORT_CENTER_Z_MM=$(python3 -c "print(-(${GROUND_FINE_MM}+${GROUND_COARSE_MM}+${SUPPORT_MM}/2))")

python3 - <<PYCHK
import math
f, c = ${GROUND_FINE_MM}, ${GROUND_COARSE_MM}
iface = f + c
seen = set()
rows = []
for e in "${EDGE_LIST}".split():
    kind, xlen, x, ylen, seed, t = e.split(":")
    t = float(t); d = iface - t
    assert t > 0, f"{e}：厚度必须 > 0"
    assert d >= f, f"{e}：空鼓上表面 {d} mm 比细泥层底 {f} mm 还浅"
    assert abs(d + t - iface) < 1e-9, f"{e}：空鼓下表面 {d+t} mm 没贴在界面 {iface} mm 上"
    assert float(x) - float(xlen)/2 == -15.0, f"{e}：边界不在 x = -15 mm"
    assert seed not in seen, f"{e}：种子重复"
    seen.add(seed)
    rows.append((t, d, f + c/2 - d - t/2, seed))
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
print(f"  粗泥层 z ∈ [{-(f+c)}, {-f}] mm，界面深 {iface} mm；各档空鼓（边界 x = -15 mm）：")
for t, d, lz, seed in rows:
    print(f"    厚 {t:>4g} mm   z ∈ [{-(d+t):g}, {-d:g}] mm   局部 z {lz:+g} mm   种子 {seed}")
print(f"  D1 占 z ∈ [{d1_bottom}, {${D1_CENTER_Z_MM}+${D1_HEIGHT_MM}/2}] mm，"
      f"下表面离物体 {d1_bottom} mm，内半径下限 {rmin_limit:.2f} mm，"
      f"实取 {${D1_RMIN_MM}} mm，余量 {${D1_RMIN_MM}-rmin_limit:+.2f} mm")
print(f"  D2 占 z ∈ [{${D2_BOTTOM_Z_MM}}, {d2_mid + ${D2_THICKNESS_MM}/2}] mm，中面 {d2_mid} mm，"
      f"源到中面 {d2_mid-src_z} mm，边缘接受半角 {accept:.1f}°，发射锥 {theta}°")
print(f"  深 30 mm 处视场半宽 {(src_z+30)*math.tan(math.radians(min(theta, accept))):.1f} mm")
PYCHK

echo "=================================================================="
echo "F3：空鼓厚度扫描（M3 构型，参照场沿用 M3）"
echo "几何      : ${MAIN_MAC} + geometry_detectors_pose_v2.mac"
echo "发射锥    : ${SOURCE_THETA_MAX_DEG}°（旧 45°）"
echo "D1        : 内半径 ${D1_RMIN_MM} / 外半径 ${D1_RMAX_MM} / 厚 ${D1_HEIGHT_MM} / 中面 z ${D1_CENTER_Z_MM} mm"
echo "D2        : 下表面 z ${D2_BOTTOM_Z_MM} / 厚 ${D2_THICKNESS_MM} mm"
echo "空鼓      : 厚 10/5/3/2/1 mm，下表面贴界面（深 ${INTERFACE_DEPTH_MM} mm）；探测头 L = ${HEAD_X_MM} mm"
echo "档数      : ${N_JOBS} 档，每档 ${N_PRIMARIES} 事例"
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
  IFS=':' read -r KIND VOID_XLEN_MM VOID_X_MM VOID_YLEN_MM RANDOM_SEED VOID_THICKNESS_MM <<< "$entry"
  VOID_Y_MM=0
  VOID_DEPTH_MM=$(python3 -c "print(${INTERFACE_DEPTH_MM}-${VOID_THICKNESS_MM})")
  VOID_LOCAL_Z_MM=$(python3 -c "print(${GROUND_FINE_MM}+${GROUND_COARSE_MM}/2-${VOID_DEPTH_MM}-${VOID_THICKNESS_MM}/2)")
  RUN_ID="m3_${KIND}_t${VOID_THICKNESS_MM}_L${HEAD_X_MM}_$((N_PRIMARIES/1000000))m_seed${RANDOM_SEED}"
  OUT_DIR="output/${RUN_ID}"
  OUTPUT_PREFIX="${OUT_DIR}/hits"

  if [ -d "${SHARE_OUT}/${RUN_ID}" ]; then
    echo "[跳过 ${JOB_N}/${N_JOBS}] 共享盘已存在 ${RUN_ID}"; echo; continue
  fi
  mkdir -p "$OUT_DIR"
  echo "=== [${JOB_N}/${N_JOBS}] ${KIND} 厚 ${VOID_THICKNESS_MM} mm（深 ${VOID_DEPTH_MM}–${INTERFACE_DEPTH_MM}）  空鼓 x ∈ [$(python3 -c "print(${VOID_X_MM}-${VOID_XLEN_MM}/2)"), $(python3 -c "print(${VOID_X_MM}+${VOID_XLEN_MM}/2)")] mm  种子 ${RANDOM_SEED}  $(date '+%T') ==="
  df -h . | tail -1

  Gate --version > "${OUT_DIR}/gate_version.txt" 2>&1 || true
  sha256sum ./*.mac ./materials/*.db > "${OUT_DIR}/macro_sha256.txt"
  {
    echo "RUN_ID=${RUN_ID}"
    echo "PURPOSE=F3 void thickness scan, M3 head geometry v2, void bottom on plaster/support interface"
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
