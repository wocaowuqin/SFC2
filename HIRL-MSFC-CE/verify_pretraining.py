def verify_pretraining():
    """验证预训练是否正确执行完整轨迹"""

    import hyperparameters as H
    from hirl_sfc_env import SFC_HIRL_Env
    from expert_msfce import parse_mat_request
    import scipy.io as sio

    print("=" * 80)
    print("预训练逻辑验证")
    print("=" * 80)

    # 1. 初始化环境
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)

    # 2. 加载第一个请求
    reqs = sio.loadmat(H.INPUT_DIR / "sorted_requests.mat")['sorted_requests']
    requests = [parse_mat_request(r) for r in reqs]
    first_req = requests[0]

    print(f"\n测试请求:")
    print(f"  ID: {first_req['id']}")
    print(f"  源: {first_req['source']}")
    print(f"  目的地数量: {len(first_req['dest'])}")
    print(f"  VNF数量: {len(first_req['vnf'])}")

    # 3. 获取专家轨迹
    network_state = env._get_network_state_dict()
    solution, expert_traj = env.expert.solve_request_for_expert(first_req, network_state)

    print(f"\n专家轨迹:")
    print(f"  轨迹长度: {len(expert_traj)}")

    if expert_traj:
        print(f"  步骤详情:")
        for i, (goal, action, cost) in enumerate(expert_traj):
            print(f"    步骤 {i + 1}: 连接目的地 {goal} (动作 {action}, 成本 {cost:.3f})")

    # 4. 模拟执行完整轨迹
    print(f"\n模拟执行:")

    # 重置环境到初始状态
    env.reset_request()
    while env.current_request is None or env.current_request['id'] != first_req['id']:
        env.reset_request()
        if env.t >= env.T:
            print("  ❌ 无法找到测试请求")
            return

    print(f"  初始未连接目的地: {env.unadded_dest_indices}")

    steps_executed = 0
    for i, (goal, action_tuple, cost) in enumerate(expert_traj):
        if goal not in env.unadded_dest_indices:
            print(f"  步骤 {i + 1}: 跳过 (目标 {goal} 已完成)")
            continue

        action = action_tuple[0] * env.K_path + action_tuple[1]
        next_state, actual_cost, sub_done, req_done = env.step_low_level(goal, action)

        steps_executed += 1
        print(f"  步骤 {i + 1}: 执行成功 | 剩余目的地: {env.unadded_dest_indices} | 请求完成: {req_done}")

        if req_done:
            print(f"\n  ✅ 请求完成! (执行 {steps_executed} 步)")
            break

    # 5. 检查结果
    print(f"\n验证结果:")
    if not env.unadded_dest_indices:
        print(f"  ✅ 成功: 所有目的地已连接")
        print(f"  ✅ served_requests_len: {len(env.served_requests)}")
    else:
        print(f"  ❌ 失败: 仍有 {len(env.unadded_dest_indices)} 个目的地未连接")

    print(f"\n结论:")
    if len(expert_traj) == steps_executed and not env.unadded_dest_indices:
        print(f"  ✅ 预训练逻辑正确: 完整执行了 {steps_executed} 步专家轨迹")
    else:
        print(f"  ⚠️ 预训练逻辑可能有问题")
        print(f"     期望执行: {len(expert_traj)} 步")
        print(f"     实际执行: {steps_executed} 步")


if __name__ == "__main__":
    verify_pretraining()