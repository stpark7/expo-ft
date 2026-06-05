#!/usr/bin/env bash
#
# base(사전학습 pi0.5) vs full(RL 학습된 EXPO) 정책을 한 번의 실행으로 평가한다.
#
# 왜 순차 실행인가:
#   * GPU가 1장(RTX 5090 32GB)이라 pi0.5를 두 번 동시에 올리면 OOM 위험.
#   * sim 클라이언트는 env_usage="eval" 키 하나로 단일 환경을 공유하므로, 두 평가가
#     동시에 붙으면 같은 env에 reset/step이 뒤섞여 서로를 오염시킨다.
#   → 그래서 base 100ep → full 100ep을 백투백으로 돌린다. 한 번만 시작해두고 자면
#     아침에 둘 다 끝나 있다. 클라이언트는 serve_forever라 두 평가 사이에도 살아 있고,
#     두 번째 평가가 접속하면 create_env로 환경을 새로 만들어 받아준다.
#
# 사전 조건: 다른 터미널에서 sim 클라이언트가 8102에 떠 있어야 한다.
#   scripts/sim/run_policy.sh
#
# 사용법(repo 루트에서):
#   scripts/sim/eval_both.sh                 # step 100000, 각 100 에피소드
#   STEP=90000 N_EPISODES=50 scripts/sim/eval_both.sh   # 덮어쓰기
#
# 둘 중 하나가 죽어도 나머지는 끝까지 돌고, 마지막에 양쪽 성공률을 요약 출력한다.

set -uo pipefail

# eval_policy.sh 가 상대경로(.venv 등)를 쓰므로 repo 루트에서 실행해야 한다.
cd "$(dirname "$0")/../.."

STEP="${STEP:-100000}"
N_EPISODES="${N_EPISODES:-100}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./log/robocasa_sim/eval_logs"
mkdir -p "$LOG_DIR"

BASE_LOG="$LOG_DIR/eval_base_step${STEP}_${TS}.log"
FULL_LOG="$LOG_DIR/eval_full_step${STEP}_${TS}.log"

echo "================================================================"
echo " EVAL BOTH  step=$STEP  episodes=$N_EPISODES (each)"
echo "   base log: $BASE_LOG"
echo "   full log: $FULL_LOG"
echo "================================================================"

# ── [1/2] base: 사전학습 pi0.5 base 정책만(residual/critic 없음) ──────────────
echo ""
echo ">>> [1/2] BASE (pretrained pi0.5, no residual/critic)  $(date)"
STEP="$STEP" N_EPISODES="$N_EPISODES" scripts/sim/eval_policy.sh base 2>&1 | tee "$BASE_LOG"
BASE_RC=${PIPESTATUS[0]}
echo ">>> [1/2] BASE finished (exit=$BASE_RC)  $(date)"

# ── [2/2] full: RL 학습된 EXPO 정책(residual + critic argmax) ─────────────────
echo ""
echo ">>> [2/2] FULL (RL-trained EXPO: residual + critic)  $(date)"
STEP="$STEP" N_EPISODES="$N_EPISODES" scripts/sim/eval_policy.sh full 2>&1 | tee "$FULL_LOG"
FULL_RC=${PIPESTATUS[0]}
echo ">>> [2/2] FULL finished (exit=$FULL_RC)  $(date)"

# ── 요약 ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " SUMMARY  step=$STEP  episodes=$N_EPISODES (each)"
echo "----------------------------------------------------------------"
if [ "$BASE_RC" -eq 0 ]; then
    echo -n "  BASE  : "; grep "^success_rate=" "$BASE_LOG" | tail -1 || echo "(success_rate 라인 없음 — 로그 확인)"
else
    echo "  BASE  : FAILED (exit=$BASE_RC) — $BASE_LOG 확인"
fi
if [ "$FULL_RC" -eq 0 ]; then
    echo -n "  FULL  : "; grep "^success_rate=" "$FULL_LOG" | tail -1 || echo "(success_rate 라인 없음 — 로그 확인)"
else
    echo "  FULL  : FAILED (exit=$FULL_RC) — $FULL_LOG 확인"
fi
echo "================================================================"
echo "지시문(객체)별 분해는 각 로그의 'Per-instruction success' 섹션 참고."
