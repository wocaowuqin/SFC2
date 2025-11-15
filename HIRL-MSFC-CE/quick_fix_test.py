def quick_fix_test():
    """验证所有修复"""

    print("🧪 快速修复验证\n")

    import hyperparameters as H
    from hirl_sfc_env import SFC_HIRL_Env

    # 测试1: 仿真时间
    print("测试 1: 检查仿真时间")
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)

    if env.T >= 721:
        print(f"  ✅ 仿真时间充足: {env.T} 步 (事件数: {len(env.events)})")
    else:
        print(f"  ⚠️ 仿真时间可能不足: {env.T} 步 (事件数: {len(env.events)})")

    # 测试2: DC覆盖率
    print("\n测试 2: 检查 DC 覆盖率")
    dc_coverage = len(env.expert.DC) / env.expert.node_num * 100
    print(f"  DC 节点: {len(env.expert.DC)}/{env.expert.node_num} ({dc_coverage:.1f}%)")

    if dc_coverage >= 70:
        print(f"  ✅ DC 覆盖率良好")
    elif dc_coverage >= 50:
        print(f"  ⚠️ DC 覆盖率一般,建议增加")
    else:
        print(f"  ❌ DC 覆盖率过低,强烈建议增加")

    # 测试3: 请求可行性
    print("\n测试 3: 检查请求可行性")
    req = env.requests[0]
    network_state = env._get_network_state_dict()

    solution, trajectory = env.expert.solve_request_for_expert(req, network_state)

    if solution and trajectory:
        print(f"  ✅ 第一个请求可求解 (轨迹长度: {len(trajectory)})")
    else:
        print(f"  ❌ 第一个请求无法求解")
        print(f"     请求详情: src={req['source']}, dests={req['dest']}, vnfs={len(req['vnf'])}")

    # 测试4: 预训练步数
    print("\n测试 4: 检查训练参数")
    print(f"  预训练步数: {H.PRE_TRAIN_STEPS}")
    print(f"  训练回合数: {H.EPISODE_LIMIT}")

    if H.PRE_TRAIN_STEPS >= 1000:
        print(f"  ✅ 预训练步数合理")
    else:
        print(f"  ⚠️ 预训练步数较少,建议增加到 1000+")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    quick_fix_test()