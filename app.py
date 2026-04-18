import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ==========================================
# 1. 舞台の構築と冷徹な数理構造
# ==========================================
st.set_page_config(page_title="Physarum Network Analytics", layout="wide")
st.title("粘菌ネットワークの最適化とレジリエンス解析")
st.markdown("視覚的ノイズを排除し、穏やかな時間軸で構造の最適化プロセスを観察します。")

@st.cache_resource
def build_base_graph():
    G = nx.hexagonal_lattice_graph(6, 6)
    pos = nx.get_node_attributes(G, 'pos')
    nodes = list(G.nodes())
    
    center_node = min(nodes, key=lambda n: (pos[n][0]-3)**2 + (pos[n][1]-5)**2)
    sorted_by_dist = sorted(nodes, key=lambda n: -((pos[n][0]-pos[center_node][0])**2 + (pos[n][1]-pos[center_node][1])**2))
    targets = [sorted_by_dist[0], sorted_by_dist[5], sorted_by_dist[18]]
    
    return G, pos, center_node, targets

G_base, pos, start_node, targets = build_base_graph()
edges = list(G_base.edges())

# ==========================================
# 2. 動的状態（State）の初期化
# ==========================================
if 'pheromones' not in st.session_state:
    st.session_state.pheromones = np.ones(len(edges)) * 0.1
if 'exploration' not in st.session_state:
    st.session_state.exploration = np.zeros(len(edges))
if 'history_cost' not in st.session_state:
    st.session_state.history_cost = []
if 'history_efficiency' not in st.session_state:
    st.session_state.history_efficiency = []
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

edge_to_idx = {edge: i for i, edge in enumerate(edges)}
edge_to_idx.update({(v, u): i for i, (u, v) in enumerate(edges)})

# ==========================================
# 3. 観測者からの介入UI（分離アーキテクチャの導入）
# ==========================================
col_ctrl, col_plot, col_metrics = st.columns([1, 2, 1.5])

with col_ctrl:
    st.header("⏳ 時間と描画の制御")
    # 描画速度と演算速度を完全に分離
    render_delay = st.slider("画面の更新間隔（秒）", 0.2, 2.0, 0.8, step=0.1, help="値を大きくするほど、画面のチカチカが抑えられ穏やかに表示されます。")
    sim_steps_per_frame = st.slider("1描画あたりの裏側での進行度", 1, 10, 3, step=1, help="画面が1回書き換わる間に、裏で何世代分の計算を進めるかを決定します。")
    
    st.markdown("---")
    st.header("🧬 環境変数")
    agents_per_epoch = st.slider("探索の熱量（個体数）", 10, 200, 80, step=10)
    decay_pheromone = st.slider("インフラ揮発率", 0.01, 0.20, 0.05, step=0.01)
    
    st.markdown("---")
    st.header("⚡ 事象の操作")
    if st.button("▶️ シミュレーション 開始/停止", use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        
    if st.button("⚡ 大規模障害の発生", use_container_width=True):
        threshold = np.percentile(st.session_state.pheromones, 70)
        strong_edges = np.where(st.session_state.pheromones > threshold)[0]
        if len(strong_edges) > 0:
            destroyed = np.random.choice(strong_edges, size=min(5, len(strong_edges)), replace=False)
            st.session_state.pheromones[destroyed] = 0.001
            st.session_state.exploration[destroyed] = 1.0
            
    if st.button("🔄 環境の初期化", use_container_width=True):
        st.session_state.pheromones = np.ones(len(edges)) * 0.1
        st.session_state.exploration = np.zeros(len(edges))
        st.session_state.history_cost = []
        st.session_state.history_efficiency = []
        st.session_state.epoch = 0
        st.session_state.is_running = False

# ==========================================
# 4. 数理生物学的演算エンジン
# ==========================================
def run_simulation_step():
    st.session_state.pheromones *= (1.0 - decay_pheromone)
    st.session_state.exploration *= 0.6
    
    successful_paths = []
    
    for _ in range(agents_per_epoch):
        current = start_node
        path = [current]
        visited = {current}
        
        for _ in range(30):
            if current in targets:
                successful_paths.append(path)
                break
                
            neighbors = [n for n in G_base.neighbors(current) if n not in visited]
            if not neighbors:
                break
                
            weights = []
            for n in neighbors:
                idx = edge_to_idx[(current, n)]
                tau = st.session_state.pheromones[idx]
                weights.append(tau ** 2.5)
                
            next_node = random.choices(neighbors, weights=weights, k=1)[0]
            
            idx = edge_to_idx[(current, next_node)]
            st.session_state.exploration[idx] += 1.0
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
    for path in successful_paths:
        reward = 10.0 / len(path)
        for u, v in zip(path[:-1], path[1:]):
            idx = edge_to_idx[(u, v)]
            st.session_state.pheromones[idx] += reward

    total_cost = np.sum(st.session_state.pheromones)
    efficiency = len(successful_paths) / agents_per_epoch * 100.0
    
    st.session_state.history_cost.append(total_cost)
    st.session_state.history_efficiency.append(efficiency)
    if len(st.session_state.history_cost) > 100:
        st.session_state.history_cost.pop(0)
        st.session_state.history_efficiency.pop(0)

# ==========================================
# 5. 構造とデータの可視化（目に優しい描画）
# ==========================================
with col_plot:
    plot_placeholder = st.empty()
    st.markdown(f"**観測世代 (Epoch):** {st.session_state.epoch}")
    
    def draw_network():
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#FAFAFA')
        ax.set_aspect('equal')
        ax.axis('off')
        
        max_p = float(np.max(st.session_state.pheromones))
        if max_p <= 0.0: max_p = 1.0
        # 探索の赤色をマイルドにするための上限調整
        max_e = float(np.max(st.session_state.exploration))
        if max_e <= 0.0: max_e = 1.0
        
        p_vals = np.clip(st.session_state.pheromones / max_p, 0, 1)
        e_vals = np.clip(st.session_state.exploration / max_e, 0, 1)
        
        edge_colors = []
        edge_widths = []
        
        for i in range(len(edges)):
            p, e = p_vals[i], e_vals[i]
            if p > 0.15:
                edge_colors.append((0.1, 0.4, 0.8, min(0.3 + p, 1.0)))
                edge_widths.append(1.0 + 5.0 * p)
            elif e > 0.05:
                # 赤色を少し彩度を落としたサーモンピンク寄りに変更し、目に優しく
                edge_colors.append((0.8, 0.4, 0.4, min(0.1 + e*1.5, 0.6)))
                edge_widths.append(0.5 + 1.5 * e)
            else:
                edge_colors.append((0.9, 0.9, 0.9, 0.4))
                edge_widths.append(0.5)
                
        nx.draw_networkx_edges(G_base, pos, ax=ax, edgelist=edges, edge_color=edge_colors, width=edge_widths)
        
        normal_nodes = list(set(G_base.nodes()) - {start_node} - set(targets))
        nx.draw_networkx_nodes(G_base, pos, nodelist=normal_nodes, ax=ax, node_size=15, node_color='#CCCCCC')
        nx.draw_networkx_nodes(G_base, pos, nodelist=[start_node], ax=ax, node_size=250, node_color='#D35400')
        nx.draw_networkx_nodes(G_base, pos, nodelist=targets, ax=ax, node_size=200, node_color='#2980B9')
        
        plot_placeholder.pyplot(fig)
        plt.close(fig)

    draw_network()

with col_metrics:
    st.subheader("リアルタイム解析データ")
    if st.session_state.history_cost:
        st.metric(label="現在のネットワーク維持コスト", value=f"{st.session_state.history_cost[-1]:.1f}")
        st.line_chart(st.session_state.history_cost, height=200, use_container_width=True)
        
        st.metric(label="ターゲット到達効率 (%)", value=f"{st.session_state.history_efficiency[-1]:.1f}%")
        st.line_chart(st.session_state.history_efficiency, height=200, use_container_width=True)

# ==========================================
# 6. 反復プロセスの実行ループ（デカップリング）
# ==========================================
if st.session_state.is_running:
    # 指定された回数分だけ、裏側で計算を進める（画面は更新しない）
    for _ in range(sim_steps_per_frame):
        run_simulation_step()
        st.session_state.epoch += 1
        
    # 指定された秒数だけ意図的に待機し、目に優しい間隔で1回だけ画面を更新する
    time.sleep(render_delay)
    st.rerun()