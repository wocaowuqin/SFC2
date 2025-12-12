import pickle
import numpy as np
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# 将当前目录添加到路径，以便导入本地模块
sys.path.append(os.getcwd())

import hyperparameters as H
from expert_msfce import MSFCE_Solver


def main():
    # ==============================
    # 1. 配置与初始化
    # ==============================
    input_dir = H.INPUT_DIR
    topo = H.TOPOLOGY_MATRIX
    dc_nodes = H.DC_NODES
    capacities = H.CAPACITIES

    print("=" * 40)
    print("  单专家算法 (MSFC-CE) 性能测试")
    print("=" * 40)

    # 1.1 加载数据
    print(f"正在从 {input_dir} 加载数据...")
    try:
        req_path = input_dir / 'sorted_requests.pkl'
        evt_path = input_dir / 'event_list.pkl'

        if not req_path.exists() or not evt_path.exists():
            print(f"错误：找不到数据文件。请确保已运行 main_generate.py 和 generate_event.py")
            return

        with open(req_path, 'rb') as f:
            requests = pickle.load(f)
        with open(evt_path, 'rb') as f:
            events = pickle.load(f)

        print(f"成功加载: {len(requests)} 条业务请求, {len(events)} 个时间步")
    except Exception as e:
        print(f"数据加载出错: {e}")
        return

    # 1.2 初始化专家求解器
    path_db_path = input_dir / "US_Backbone_path.mat"
    if not path_db_path.exists():
        # 尝试在当前目录查找
        path_db_path = Path("US_Backbone_path.mat")

    if not path_db_path.exists():
        print(f"错误：找不到路径数据库文件 (US_Backbone_path.mat)")
        return

    print(f"正在初始化专家算法...")
    expert = MSFCE_Solver(path_db_path, topo, dc_nodes, capacities)

    # 1.3 初始化网络资源状态
    node_num = topo.shape[0]
    link_num = expert.link_num  # 从专家对象获取链路数

    # 这里的状态字典必须与 expert_msfce.py 期望的格式一致
    current_state = {
        'cpu': np.full(node_num, capacities['cpu'], dtype=float),
        'mem': np.full(node_num, capacities['memory'], dtype=float),
        'bw': np.full(link_num, capacities['bandwidth'], dtype=float),
        'hvt': np.zeros((node_num, expert.type_num), dtype=int),
        'bw_ref_count': np.zeros(link_num, dtype=int)  # 辅助计数，专家算法本身可能不强依赖，但为了完整性保留
    }

    # 用于跟踪活跃业务以释放资源: {req_id: {'resources': result_tree_copy, 'req_data': req}}
    active_requests = {}

    # 统计指标
    stats = {
        'total': 0,
        'accepted': 0,
        'blocked': 0,
        'start_time': time.time()
    }

    # ==============================
    # 2. 仿真主循环
    # ==============================
    # 预处理事件列表格式，防止键名不匹配
    formatted_events = []
    for evt in events:
        # 兼容 arrive/arrive_event 键名
        arr = evt.get('arrive', evt.get('arrive_event', []))
        # 兼容 leave/leave_event 键名
        lev = evt.get('leave', evt.get('leave_event', []))
        formatted_events.append({'arrive': arr, 'leave': lev})

    # 建立 ID 到请求数据的映射
    req_map = {r['id']: r for r in requests}
    T = len(formatted_events)

    print(f"开始仿真 (总时间步: {T})...")

    for t in range(T):
        if t % 50 == 0:
            acc_rate = stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
            print(
                f"Step {t}/{T} | Req: {stats['total']} | Acc: {stats['accepted']} | Blk: {stats['blocked']} | Rate: {acc_rate:.2%}")

        # ----------------------
        # A. 处理业务离开 (释放资源)
        # ----------------------
        leave_ids = formatted_events[t]['leave']
        # 确保 leave_ids 是列表
        if isinstance(leave_ids, (np.ndarray, np.generic)):
            leave_ids = leave_ids.flatten().tolist()
        elif not isinstance(leave_ids, list):
            leave_ids = [leave_ids]

        for r_id in leave_ids:
            # 注意：某些请求可能因为之前被阻塞而不在 active_requests 中
            if r_id in active_requests:
                info = active_requests[r_id]
                res_tree = info['resources']
                req = info['req_data']

                # 1. 释放带宽 (Tree 向量: 0/1)
                bw_val = req['bw_origin']
                current_state['bw'] += res_tree['tree'] * bw_val
                # 修正浮点误差，不超过上限
                np.minimum(current_state['bw'], capacities['bandwidth'], out=current_state['bw'])

                # 2. 释放节点资源 (CPU/Mem)
                # 利用 HVT 矩阵 (Node x VNF_Type)
                # 遍历所有节点
                for n_idx in range(node_num):
                    # 遍历该请求在该节点部署的所有 VNF 类型
                    # res_tree['hvt'][n_idx] 是一个行向量
                    deployed_types = np.where(res_tree['hvt'][n_idx] > 0)[0]

                    for v_type_idx in deployed_types:
                        count = int(res_tree['hvt'][n_idx, v_type_idx])
                        if count <= 0: continue

                        # 找到该类型对应的 CPU/Mem 需求
                        # 请求中的 vnf 列表存储的是 type_id (通常是 1-8)，而 v_type_idx 是 0-7
                        target_type = v_type_idx + 1

                        # 查找请求中所有匹配该类型的 VNF 索引
                        # 因为一个请求可能包含多个相同类型的 VNF
                        # 这里我们需要精确知道释放多少。由于 HVT 只是计数，
                        # 我们假设同类型的 VNF 需求相同（或者在此简化处理：
                        # 在 generate_data 中，同一次生成的请求里，相同类型的系数是固定的吗？
                        # 是的，generate_vnfs_catalog 全局生成一次，所以特定 Type 的需求系数是固定的）
                        # 因此：释放量 = count * (bw_origin * coef_of_type)
                        # 我们可以直接从 request 的 cpu_origin 列表中找到对应类型的需求值。

                        # 为了安全，我们在请求的 vnf 列表中找到对应类型的第一个需求的数值
                        # (假设同类型需求一致，或者平均一致)
                        try:
                            v_list_idx = req['vnf'].index(target_type)
                            cpu_req = req['cpu_origin'][v_list_idx]
                            mem_req = req['memory_origin'][v_list_idx]

                            current_state['cpu'][n_idx] += count * cpu_req
                            current_state['mem'][n_idx] += count * mem_req

                            # 更新状态 HVT
                            current_state['hvt'][n_idx, v_type_idx] = max(0, current_state['hvt'][
                                n_idx, v_type_idx] - count)
                        except ValueError:
                            pass  # 该类型不在请求列表中? 理论上不应发生

                # 修正上限
                np.minimum(current_state['cpu'], capacities['cpu'], out=current_state['cpu'])
                np.minimum(current_state['mem'], capacities['memory'], out=current_state['mem'])

                del active_requests[r_id]

        # ----------------------
        # B. 处理业务到达 (专家决策)
        # ----------------------
        arrive_ids = formatted_events[t]['arrive']
        if isinstance(arrive_ids, (np.ndarray, np.generic)):
            arrive_ids = arrive_ids.flatten().tolist()
        elif not isinstance(arrive_ids, list):
            arrive_ids = [arrive_ids]

        for r_id in arrive_ids:
            if r_id not in req_map: continue

            stats['total'] += 1
            req = req_map[r_id]

            # --- 核心：调用专家算法 ---
            # solve_request_for_expert 会返回 (tree, traj)
            # 如果失败返回 (None, [])
            res_tree, _ = expert.solve_request_for_expert(req, current_state)

            if res_tree is not None:
                stats['accepted'] += 1

                # 更新资源状态 (扣除)
                # 1. 扣带宽
                bw_val = req['bw_origin']
                current_state['bw'] = np.maximum(0, current_state['bw'] - res_tree['tree'] * bw_val)

                # 2. 扣节点资源
                for n_idx in range(node_num):
                    deployed_types = np.where(res_tree['hvt'][n_idx] > 0)[0]
                    for v_type_idx in deployed_types:
                        count = int(res_tree['hvt'][n_idx, v_type_idx])
                        target_type = v_type_idx + 1
                        try:
                            v_list_idx = req['vnf'].index(target_type)
                            cpu_req = req['cpu_origin'][v_list_idx]
                            mem_req = req['memory_origin'][v_list_idx]

                            current_state['cpu'][n_idx] = max(0, current_state['cpu'][n_idx] - count * cpu_req)
                            current_state['mem'][n_idx] = max(0, current_state['mem'][n_idx] - count * mem_req)
                            current_state['hvt'][n_idx, v_type_idx] += count
                        except ValueError:
                            pass

                # 记录以备释放
                active_requests[r_id] = {'resources': res_tree, 'req_data': req}
            else:
                stats['blocked'] += 1

    # ==============================
    # 3. 输出最终结果
    # ==============================
    elapsed = time.time() - stats['start_time']
    acc_rate = stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
    blk_rate = stats['blocked'] / stats['total'] if stats['total'] > 0 else 0

    print("\n" + "#" * 50)
    print("           测试结果报告")
    print("#" * 50)
    print(f"总耗时:          {elapsed:.2f} 秒")
    print(f"总请求数:        {stats['total']}")
    print(f"接受数 (Accepted): {stats['accepted']}")
    print(f"阻塞数 (Blocked):  {stats['blocked']}")
    print("-" * 30)
    print(f"接受率 (Acceptance Rate): {acc_rate:.2%}")
    print(f"阻塞率 (Blocking Rate):   {blk_rate:.2%}")
    print("#" * 50)


if __name__ == "__main__":
    main()