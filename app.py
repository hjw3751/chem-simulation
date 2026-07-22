import matplotlib.pyplot as plt
import streamlit as st

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="화학 평형 RK4 시뮬레이터", page_icon="🧪", layout="wide"
)


# ==========================================
# 1. RK4(Runge-Kutta 4th Order) 수치해석 엔진
# ==========================================
def rk4_step(funcs, t, y, dt, params):
    """4차 룽게-쿠타 수치해석 알고리즘 하드코딩"""
    k1 = funcs(t, y, params)
    y_k1 = [y[i] + 0.5 * dt * k1[i] for i in range(len(y))]

    k2 = funcs(t + 0.5 * dt, y_k1, params)
    y_k2 = [y[i] + 0.5 * dt * k2[i] for i in range(len(y))]

    k3 = funcs(t + 0.5 * dt, y_k3 := y_k2, params)
    y_k3 = [y[i] + dt * k3[i] for i in range(len(y))]

    k4 = funcs(t + dt, y_k3, params)

    return [
        y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(len(y))
    ]


def chemical_rates(t, y, params):
    """가역 반응 미분방정식 (A + B <-> C + D)"""
    A, B, C, D = y
    k_f, k_b = params
    r_f = k_f * A * B
    r_b = k_b * C * D

    dA_dt = -r_f + r_b
    dB_dt = -r_f + r_b
    dC_dt = r_f - r_b
    dD_dt = r_f - r_b
    return [dA_dt, dB_dt, dC_dt, dD_dt]


# ==========================================
# 2. 메인 화면 UI 및 사용법 가이드
# ==========================================
st.title("🧪 화학 평형 & 르샤틀리에 시뮬레이터")

# 초보자를 위한 친절한 설명 박스
with st.expander("📖 처음 오셨나요? 시뮬레이터 사용법 & 화학 개념 설명", expanded=True):
    st.markdown("""
    ### 💡 이 시뮬레이터는 무엇인가요?
    가역 반응 **A + B ⇌ C + D** 에서 조건이 변할 때 **동적 평형 상태**와 **르샤틀리에 원리**에 의해 반응이 어느 방향으로 이동하는지 수치해석 알고리즘(RK4)으로 실시간 계산하여 시각화하는 웹 도구입니다.

    ---
    ### ⚙️ 어떻게 사용하나요?
    1. **왼쪽 사이드바**에서 정반응/역반응 속도상수와 초기 농도를 조절해 보세요.
    2. **20초 시점 자극 설정**에서 반응물 A나 생성물 C의 농도를 늘리거나 줄여보세요.
    3. 그래프가 실시간으로 변하며 **Q와 K의 비교** 및 **새로운 평형 도달 과정**을 보여줍니다.
    
    ---
    ### 🔍 주요 용어 쉬운 정리
    * **평형 상수 (K)**: 반응이 평형에 도달했을 때 반응물과 생성물의 농도 비율입니다. ($K = k_f / k_b$)
    * **반응 지수 (Q)**: 임의의 순간에 존재하는 반응물과 생성물의 농도 비율입니다.
    * **르샤틀리에 원리**: 평형 상태의 계에 외부 자극(농도 변화 등)이 가해지면, 그 자극을 완화하려는 방향으로 평형이 이동한다는 법칙입니다.
      * **Q < K** : 정반응 진행 (생성물 쪽으로 이동)
      * **Q > K** : 역반응 진행 (반응물 쪽으로 이동)
      * **Q = K** : 평형 유지
    """)

st.divider()

# ==========================================
# 3. 사이드바 - 사용자 입력 컨트롤러
# ==========================================
st.sidebar.header("🎛️ 시뮬레이션 매개변수 설정")

st.sidebar.subheader("1. 속도 상수 (Reaction Rates)")
k_f = st.sidebar.slider(
    "정반응 속도 상수 (k_f)",
    min_value=0.01,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="정반응(A+B -> C+D)이 얼마나 빠르게 일어나는지를 결정합니다. 값이 커질수록 정반응이 우세해집니다.",
)
k_b = st.sidebar.slider(
    "역반응 속도 상수 (k_b)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="역반응(C+D -> A+B)이 얼마나 빠르게 일어나는지를 결정합니다. 평형 상수 K는 (k_f / k_b)로 계산됩니다.",
)

K_eq = k_f / k_b

st.sidebar.subheader("2. 초기 농도 설정 (M)")
A0 = st.sidebar.number_input(
    "[A] 초기 농도",
    min_value=0.0,
    value=2.0,
    step=0.1,
    help="반응 시작 시점(t=0s)에 존재하는 반응물 A의 몰농도(M)입니다.",
)
B0 = st.sidebar.number_input(
    "[B] 초기 농도",
    min_value=0.0,
    value=2.0,
    step=0.1,
    help="반응 시작 시점(t=0s)에 존재하는 반응물 B의 몰농도(M)입니다.",
)
C0 = st.sidebar.number_input(
    "[C] 초기 농도",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="반응 시작 시점(t=0s)에 존재하는 생성물 C의 몰농도(M)입니다.",
)
D0 = st.sidebar.number_input(
    "[D] 초기 농도",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="반응 시작 시점(t=0s)에 존재하는 생성물 D의 몰농도(M)입니다.",
)

st.sidebar.subheader("3. 20초 시점 외부 자극 (농도 변화)")
delta_A = st.sidebar.number_input(
    "[A] 농도 변화량 (M)",
    value=1.5,
    step=0.1,
    help="t=20초 순간에 [A]를 얼마나 추가(+)하거나 제거(-)할지 설정합니다. 양수는 농도 증가, 음수는 농도 감소를 의미합니다.",
)
delta_C = st.sidebar.number_input(
    "[C] 농도 변화량 (M)",
    value=0.0,
    step=0.1,
    help="t=20초 순간에 [C]를 얼마나 추가(+)하거나 제거(-)할지 설정합니다. 생성물 변화에 따른 역반응/정반응 이동을 관찰할 수 있습니다.",
)


# ==========================================
# 4. 수치해석 연산 및 결과 데이터 생성
# ==========================================
dt = 0.05
total_steps = 800

# 1차 평형 연산 (0~20s)
y = [A0, B0, C0, D0]
history_t = [0]
history_y = [[y[i]] for i in range(4)]

for step in range(400):
    t = step * dt
    y = rk4_step(chemical_rates, t, y, dt, (k_f, k_b))
    history_t.append(t + dt)
    for i in range(4):
        history_y[i].append(y[i])

eq1_A, eq1_B, eq1_C, eq1_D = y
Q1 = (
    (eq1_C * eq1_D) / (eq1_A * eq1_B)
    if (eq1_A * eq1_B) > 0
    else float("inf")
)

# 20초 시점 자극 적용
y[0] = max(0.0001, y[0] + delta_A)
y[2] = max(0.0001, y[2] + delta_C)

# 자극 직후 Q 값 및 예측
Q_pert = (y[2] * y[3]) / (y[0] * y[1]) if (y[0] * y[1]) > 0 else float("inf")

if Q_pert < K_eq - 1e-4:
    direction_text = "정반응 진행 (Forward Reaction)"
    pred_code = "Q < K"
elif Q_pert > K_eq + 1e-4:
    direction_text = "역반응 진행 (Reverse Reaction)"
    pred_code = "Q > K"
else:
    direction_text = "평형 유지 (Equilibrium Shift N/A)"
    pred_code = "Q ≈ K"

# 2차 재평형 연산 (20~40s)
for step in range(400, total_steps):
    t = step * dt
    y = rk4_step(chemical_rates, t, y, dt, (k_f, k_b))
    history_t.append(t + dt)
    for i in range(4):
        history_y[i].append(y[i])


# ==========================================
# 5. 결과 대시보드 지표 및 영문 그래프 출력
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("평형 상수 (K)", f"{K_eq:.4f}")
col2.metric("1차 평형 Q 값", f"{Q1:.4f}")
col3.metric("자극 직후 Q 값", f"{Q_pert:.4f}")
col4.metric("르샤틀리에 이동 예측", direction_text)

# 표준 영문 그래프 시각화 (폰트 깨짐 완벽 방지)
fig, ax = plt.subplots(figsize=(10, 5))

labels_en = [
    "Reactant [A]",
    "Reactant [B]",
    "Product [C]",
    "Product [D]",
]
colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

for i in range(4):
    ax.plot(
        history_t,
        history_y[i],
        label=labels_en[i],
        color=colors[i],
        linewidth=2.5,
    )

ax.axvline(
    x=20,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="External Perturbation (t=20s)",
)

ax.set_title(
    f"RK4 Chemical Equilibrium Simulation [{pred_code}]",
    fontsize=14,
    pad=10,
)
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("Concentration (M)", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=10, loc="upper right")

st.pyplot(fig)

# 그래프 하단 영문 용어 한글 해설 가이드
st.caption("""
💡 **그래프 범례 및 축 설명 (Graph Legend Guide)**
* **Reactant [A], [B]**: 반응물 A와 B의 농도 변화 (Molar Concentration)
* **Product [C], [D]**: 생성물 C와 D의 농도 변화
* **External Perturbation (t=20s)**: 20초 시점에 가해진 외부 농도 자극 투입 시점
* **Time (seconds) / Concentration (M)**: 시간(초) 및 몰농도(M)
""")
