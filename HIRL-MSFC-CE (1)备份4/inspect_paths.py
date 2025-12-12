import scipy.io as sio
import numpy as np
import sys

# === 导入你的项目文件 ===
try:
    import hyperparameters as H
    from expert_msfce import MSFCE_Solver, parse_mat_request
except ImportError:
    print("错误: 找不到 hyperparameters.py 或 expert_msfce.py。")
    print("请将此脚本与这两个文件放在同一个文件夹中再运行。")
    sys.exit()


def check_paths():
    print("--- 正在检查 'sorted_requests.mat' 的路径与 'DC_NODES' 的匹配性 ---")

    # 1. 加载专家
    try:
        # 专家需要所有这些参数来初始化
        expert = MSFCE_Solver(
            H.INPUT_DIR / "US_Backbone_path.mat",
            H.TOPOLOGY_MATRIX,
            H.DC_NODES,
            H.CAPACITIES
        )
        dc_set = expert.DC
        k_paths_count = expert.k_path_count
        print(f"✅ 专家已加载 ({k_paths_count} K-paths, {len(dc_set)} DC 节点)\n")
    except Exception as e:
        print(f"错误: 无法初始化 MSFCE_Solver: {e}")
        return

    # 2. 加载请求
    try:
        reqs_path = H.INPUT_DIR / "sorted_requests.mat"
        reqs_mat = sio.loadmat(reqs_path)['sorted_requests']
        requests = [parse_mat_request(r) for r in reqs_mat]
        print(f"成功加载 {len(requests)} 个请求 (来自 {reqs_path})\n")
    except Exception as e:
        print(f"错误: 加载 'sorted_requests.mat' 失败: {e}")
        return

    # 3. 检查前 20 个请求
    print("--- 检查前 20 个请求的 K-Shortest-Paths ---")

    total_requests_failed = 0

    for i, req in enumerate(requests[:20]):
        req_id = req['id']
        source_node = req['source']

        # 我们只检查第一个目的地 (d_idx = 0)
        # 这与 _calc_eval 中的逻辑一致
        if not req['dest']:
            print(f"请求 #{req_id}: 跳过 (没有目的地)")
            continue

        dest_node = req['dest'][0]
        vnf_needed = len(req['vnf'])

        print(f"\n请求 #{req_id} (S={source_node}, D={dest_node}): 需要 {vnf_needed} 个 VNF (DC 节点)")

        is_request_feasible = False

        for k in range(1, k_paths_count + 1):
            try:
                # 从 .mat 数据库获取路径节点
                path_nodes, _, _ = expert._get_path_from_db(source_node, dest_node, k)
                if not path_nodes:
                    print(f"  Path k={k}: 未找到路径")
                    continue

                # 检查路径上有多少 DC 节点
                usable_dc_on_path = [n for n in path_nodes if n in dc_set]
                dc_count = len(usable_dc_on_path)

                if dc_count >= vnf_needed:
                    print(f"  Path k={k}: 发现 {dc_count} 个 DC 节点 (需要 {vnf_needed}) -> ✅ OK")
                    is_request_feasible = True
                else:
                    print(f"  Path k={k}: 发现 {dc_count} 个 DC 节点 (需要 {vnf_needed}) -> ❌ 失败")

            except Exception as e:
                print(f"  Path k={k}: 检查时出错: {e}")

        if not is_request_feasible:
            print(f"  -> 结论: ‼️ 请求 #{req_id} 失败 (所有 {k_paths_count} 条 K-路径都没有足够的 DC 节点)")
            total_requests_failed += 1

    print("\n--- 诊断摘要 ---")
    if total_requests_failed > 0:
        print(f"❌ 致命错误: 在检查的 20 个请求中, 有 {total_requests_failed} 个请求失败了。")
        print("因为它们的所有 K-最短路径上的 DC 节点数 < 它们所需的 VNF 数。")
        print("这就是你的专家返回 '空轨迹' 并且预训练 0 步的原因。")
        print("\n请执行以下操作之一:")
        print("1. (推荐) 重新生成你的 'US_Backbone_path.mat' 文件，使用不同的 K-最短路径算法（例如，考虑 DC 节点的算法）。")
        print("2. (简单) 修改 'hyperparameters.py' 中的 'DC_NODES' 列表，添加更多沿途的节点。")
        print("3. (简单) 重新生成 'sorted_requests.mat'，使请求的 VNF 数量减少 (例如，len(req['vnf']) 始终为 1 或 2)。")
    else:
        print("✅ 诊断通过。所有检查的请求至少有一条 K-路径拥有足够的 DC 节点。")
        print("问题更加微妙。")


if __name__ == "__main__":
    check_paths()