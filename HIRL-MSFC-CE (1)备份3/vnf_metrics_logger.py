# vnf_metrics_logger.py
"""
VNF部署指标收集与CSV导出工具 (最终融合版)
集成功能：
1. 详细资源消耗统计（去重后的实际CPU/MEM/BW消耗）
2. 部署成功率、阻塞率统计
3. 支持按时间步 (time_step) 记录与聚合，用于绘制随时间变化的趋势图
"""
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json


class VNFMetricsLogger:
    """
    记录VNF部署过程中的详细指标
    """

    def __init__(self, network_info: dict):
        """
        Args:
            network_info: {
                "total_nodes": int,
                "total_cpu": float,
                "total_bw": float,
                "total_mem": float,
                # 可选：单节点/链路容量，用于更精确计算利用率
                "node_cpu_capacity": float,
                "node_mem_capacity": float,
                "link_bw_capacity": float
            }
        """
        self.network_info = network_info

        # ====== 单次部署指标 (Current Deployment Context) ======
        self.current_deployment = {
            "time_step": 0,  # [新增] 时间步
            "request_id": None,
            "start_time": None,
            "vnf_chain": [],
            "destinations": [],

            # 资源消耗（累计所有路径 - 用于统计总消耗）
            "total_cpu_consumed": 0.0,
            "total_bw_consumed": 0.0,
            "total_mem_consumed": 0.0,

            # 资源消耗（按节点/链路去重统计 - 用于计算精确利用率）
            "node_cpu_usage": {},  # {node_id: cpu_used}
            "node_mem_usage": {},  # {node_id: mem_used}
            "link_bw_usage": {},  # {(node1, node2): bw_used}

            # 实际使用的节点和链路集合
            "used_nodes": set(),
            "used_links": set(),

            # 部署结果
            "fully_deployed": False,
            "partial_deployed": False,
            "destinations_connected": 0,
            "destinations_failed": 0,

            # 失败节点信息
            "failed_nodes": [],
            "failure_reasons": [],

            # 路径信息
            "total_hops": 0,
            "paths": [],

            # 备份策略使用
            "backup_used": False,
            "backup_levels": [],

            # 耗时
            "deployment_time": 0.0
        }

        # ====== 全局统计 (Global Stats) ======
        self.global_stats = {
            "total_requests": 0,
            "fully_accepted": 0,
            "partially_accepted": 0,
            "totally_blocked": 0,

            "total_cpu_consumed": 0.0,
            "total_bw_consumed": 0.0,
            "total_mem_consumed": 0.0,

            "avg_cpu_utilization": [],
            "avg_bw_utilization": [],
            "avg_mem_utilization": [],

            # 失败节点统计
            "failed_nodes_count": defaultdict(int),
            "failure_reasons_count": defaultdict(int),

            # 备份策略统计
            "backup_usage_count": defaultdict(int),
        }

        # ====== 时序记录 (History) ======
        self.deployment_history = []
        self.resource_utilization_history = []

    # ================================================================
    # 📊 记录单次部署
    # ================================================================

    def start_deployment(self, request_id: str, vnf_chain: List[str],
                         destinations: List[int], t: int = 0):  # <--- [核心修改] 增加 t 参数
        """开始记录一次部署，传入当前时间步 t"""
        self.current_deployment = {
            "time_step": t,  # <--- 记录时间步
            "request_id": request_id,
            "start_time": datetime.now(),
            "vnf_chain": vnf_chain,
            "destinations": destinations,

            # 初始化资源统计容器
            "total_cpu_consumed": 0.0,
            "total_bw_consumed": 0.0,
            "total_mem_consumed": 0.0,
            "node_cpu_usage": {},
            "node_mem_usage": {},
            "link_bw_usage": {},
            "used_nodes": set(),
            "used_links": set(),

            # 初始化结果状态
            "fully_deployed": False,
            "partial_deployed": False,
            "destinations_connected": 0,
            "destinations_failed": 0,
            "failed_nodes": [],
            "failure_reasons": [],
            "total_hops": 0,
            "paths": [],
            "backup_used": False,
            "backup_levels": [],
            "deployment_time": 0.0
        }

    def record_step(self,
                    step_info: dict,
                    resource_consumed: dict,
                    network_state: dict):
        """
        记录每一步的部署信息 (集成详细资源计算逻辑)
        """
        # 1. 累计总消耗（路径级累加，可能重复计算共享节点）
        self.current_deployment["total_cpu_consumed"] += resource_consumed.get("cpu", 0)
        self.current_deployment["total_bw_consumed"] += resource_consumed.get("bw", 0)
        self.current_deployment["total_mem_consumed"] += resource_consumed.get("mem", 0)

        # 2. 记录节点级资源使用（去重逻辑）
        if step_info.get("success") and "vnf_placement" in step_info:
            for node_id, resources in step_info["vnf_placement"].items():
                self.current_deployment["used_nodes"].add(node_id)
                # 累加同一节点的资源（如果多个VNF部署在同一节点）
                self.current_deployment["node_cpu_usage"][node_id] = \
                    self.current_deployment["node_cpu_usage"].get(node_id, 0) + resources.get("cpu", 0)
                self.current_deployment["node_mem_usage"][node_id] = \
                    self.current_deployment["node_mem_usage"].get(node_id, 0) + resources.get("mem", 0)

        # 3. 记录链路级资源使用（去重逻辑）
        if step_info.get("success") and "link_usage" in step_info:
            for link, resources in step_info["link_usage"].items():
                self.current_deployment["used_links"].add(link)
                # 链路资源取最大值（多播共享带宽特性）
                current_bw = self.current_deployment["link_bw_usage"].get(link, 0)
                self.current_deployment["link_bw_usage"][link] = \
                    max(current_bw, resources.get("bw", 0))

        # 4. 记录路径与结果
        if step_info.get("success"):
            self.current_deployment["destinations_connected"] += 1
            self.current_deployment["paths"].append(step_info.get("path", []))
            self.current_deployment["total_hops"] += len(step_info.get("path", [])) - 1
        else:
            self.current_deployment["destinations_failed"] += 1
            self.current_deployment["failed_nodes"].append(step_info.get("destination"))
            self.current_deployment["failure_reasons"].append(
                step_info.get("failure_reason", "unknown")
            )

        # 5. 记录备份策略使用
        if step_info.get("backup_used"):
            self.current_deployment["backup_used"] = True
            self.current_deployment["backup_levels"].append(
                step_info.get("backup_level", "unknown")
            )

        # 6. 记录资源利用率（使用去重后的实际消耗）
        actual_cpu_used = sum(self.current_deployment["node_cpu_usage"].values())
        actual_mem_used = sum(self.current_deployment["node_mem_usage"].values())
        actual_bw_used = sum(self.current_deployment["link_bw_usage"].values())

        num_used_nodes = len(self.current_deployment["used_nodes"])
        num_used_links = len(self.current_deployment["used_links"])

        # 计算利用率（只统计使用的节点/链路）
        if num_used_nodes > 0:
            node_capacity = network_state.get("node_cpu_capacity", 80.0)  # 单节点容量默认值
            cpu_util = actual_cpu_used / (num_used_nodes * node_capacity)
            mem_util = actual_mem_used / (num_used_nodes * network_state.get("node_mem_capacity", 60.0))
        else:
            cpu_util = 0.0
            mem_util = 0.0

        if num_used_links > 0:
            link_capacity = network_state.get("link_bw_capacity", 80.0)
            bw_util = actual_bw_used / (num_used_links * link_capacity)
        else:
            bw_util = 0.0

        self.resource_utilization_history.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": self.current_deployment["request_id"],
            "cpu_utilization": cpu_util,
            "bw_utilization": bw_util,
            "mem_utilization": mem_util,
            "num_used_nodes": num_used_nodes,
            "num_used_links": num_used_links
        })

    def end_deployment(self, network_state: dict):
        """结束一次部署的记录"""
        # 计算部署时间
        if self.current_deployment["start_time"]:
            elapsed = datetime.now() - self.current_deployment["start_time"]
            self.current_deployment["deployment_time"] = elapsed.total_seconds()

        # 判断部署结果
        total_dest = len(self.current_deployment["destinations"])
        connected = self.current_deployment["destinations_connected"]

        if connected == total_dest:
            self.current_deployment["fully_deployed"] = True
            self.global_stats["fully_accepted"] += 1
        elif connected > 0:
            self.current_deployment["partial_deployed"] = True
            self.global_stats["partially_accepted"] += 1
        else:
            self.global_stats["totally_blocked"] += 1

        # 更新全局统计
        self.global_stats["total_requests"] += 1
        self.global_stats["total_cpu_consumed"] += self.current_deployment["total_cpu_consumed"]
        self.global_stats["total_bw_consumed"] += self.current_deployment["total_bw_consumed"]
        self.global_stats["total_mem_consumed"] += self.current_deployment["total_mem_consumed"]

        # 计算本次部署的实际资源利用率（去重后）
        actual_cpu = sum(self.current_deployment["node_cpu_usage"].values())
        actual_mem = sum(self.current_deployment["node_mem_usage"].values())
        actual_bw = sum(self.current_deployment["link_bw_usage"].values())

        num_nodes = len(self.current_deployment["used_nodes"])
        num_links = len(self.current_deployment["used_links"])

        node_capacity = network_state.get("node_cpu_capacity", 80.0)
        mem_capacity = network_state.get("node_mem_capacity", 60.0)
        link_capacity = network_state.get("link_bw_capacity", 80.0)

        if num_nodes > 0:
            cpu_util = actual_cpu / (num_nodes * node_capacity)
            mem_util = actual_mem / (num_nodes * mem_capacity)
        else:
            cpu_util = 0.0
            mem_util = 0.0

        if num_links > 0:
            bw_util = actual_bw / (num_links * link_capacity)
        else:
            bw_util = 0.0

        self.global_stats["avg_cpu_utilization"].append(cpu_util)
        self.global_stats["avg_bw_utilization"].append(bw_util)
        self.global_stats["avg_mem_utilization"].append(mem_util)

        # 记录失败节点和原因
        for node in self.current_deployment["failed_nodes"]:
            self.global_stats["failed_nodes_count"][node] += 1
        for reason in self.current_deployment["failure_reasons"]:
            self.global_stats["failure_reasons_count"][reason] += 1

        # 记录备份策略使用
        for level in self.current_deployment["backup_levels"]:
            self.global_stats["backup_usage_count"][level] += 1

        # 保存到历史记录
        self.deployment_history.append(self.current_deployment.copy())

    # ================================================================
    # 📈 计算统计指标
    # ================================================================

    def compute_statistics(self) -> dict:
        """计算汇总统计"""
        total = self.global_stats["total_requests"]
        if total == 0:
            return {}

        stats = {
            # 接受率
            "full_acceptance_rate": self.global_stats["fully_accepted"] / total,
            "partial_acceptance_rate": self.global_stats["partially_accepted"] / total,
            "blocking_rate": self.global_stats["totally_blocked"] / total,

            # 平均资源消耗（每次部署）
            "avg_cpu_per_deployment": self.global_stats["total_cpu_consumed"] / total,
            "avg_bw_per_deployment": self.global_stats["total_bw_consumed"] / total,
            "avg_mem_per_deployment": self.global_stats["total_mem_consumed"] / total,

            # 平均资源利用率
            "avg_cpu_utilization": np.mean(self.global_stats["avg_cpu_utilization"]),
            "avg_bw_utilization": np.mean(self.global_stats["avg_bw_utilization"]),
            "avg_mem_utilization": np.mean(self.global_stats["avg_mem_utilization"]),

            # 资源利用率标准差
            "std_cpu_utilization": np.std(self.global_stats["avg_cpu_utilization"]),
            "std_bw_utilization": np.std(self.global_stats["avg_bw_utilization"]),
            "std_mem_utilization": np.std(self.global_stats["avg_mem_utilization"]),

            # 失败节点分析
            "top_failed_nodes": sorted(
                self.global_stats["failed_nodes_count"].items(),
                key=lambda x: x[1], reverse=True
            )[:10],

            "failure_reasons": dict(self.global_stats["failure_reasons_count"]),

            # 备份策略使用统计
            "backup_usage": dict(self.global_stats["backup_usage_count"]),

            # 平均跳数
            "avg_hops": np.mean([
                d["total_hops"] / max(d["destinations_connected"], 1)
                for d in self.deployment_history
            ]),

            # 平均部署时间
            "avg_deployment_time": np.mean([
                d["deployment_time"] for d in self.deployment_history
            ])
        }

        return stats

    # ================================================================
    # 💾 CSV导出功能
    # ================================================================

    def export_deployment_details(self, filename: str = "deployment_details.csv"):
        """导出每次部署的详细信息 (包含时间步)"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            # [核心修改] 增加 time_step 列
            fieldnames = [
                'time_step', 'request_id', 'vnf_chain', 'total_destinations',
                'destinations_connected', 'destinations_failed',
                'total_cpu_consumed', 'total_bw_consumed', 'total_mem_consumed',
                'actual_cpu_used', 'actual_bw_used', 'actual_mem_used',
                'num_used_nodes', 'num_used_links',
                'avg_cpu_util_per_node', 'avg_bw_util_per_link', 'avg_mem_util_per_node',
                'total_hops', 'avg_hops_per_dest',
                'fully_deployed', 'partial_deployed',
                'backup_used', 'backup_levels',
                'failed_nodes', 'failure_reasons',
                'deployment_time'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for dep in self.deployment_history:
                avg_hops = dep["total_hops"] / max(dep["destinations_connected"], 1)

                # 计算去重后的实际资源使用 (用于导出)
                actual_cpu = sum(dep["node_cpu_usage"].values())
                actual_mem = sum(dep["node_mem_usage"].values())
                actual_bw = sum(dep["link_bw_usage"].values())

                num_nodes = len(dep["used_nodes"])
                num_links = len(dep["used_links"])

                node_capacity = 80.0  # 默认值，或者从 self.network_info 读取
                mem_capacity = 60.0
                link_capacity = 80.0

                cpu_util = (actual_cpu / (num_nodes * node_capacity)) if num_nodes > 0 else 0.0
                mem_util = (actual_mem / (num_nodes * mem_capacity)) if num_nodes > 0 else 0.0
                bw_util = (actual_bw / (num_links * link_capacity)) if num_links > 0 else 0.0

                writer.writerow({
                    'time_step': dep.get('time_step', 0),  # [修改] 写入时间步
                    'request_id': dep['request_id'],
                    'vnf_chain': '->'.join(dep['vnf_chain']),
                    'total_destinations': len(dep['destinations']),
                    'destinations_connected': dep['destinations_connected'],
                    'destinations_failed': dep['destinations_failed'],
                    'total_cpu_consumed': f"{dep['total_cpu_consumed']:.4f}",
                    'total_bw_consumed': f"{dep['total_bw_consumed']:.4f}",
                    'total_mem_consumed': f"{dep['total_mem_consumed']:.4f}",
                    'actual_cpu_used': f"{actual_cpu:.4f}",
                    'actual_bw_used': f"{actual_bw:.4f}",
                    'actual_mem_used': f"{actual_mem:.4f}",
                    'num_used_nodes': num_nodes,
                    'num_used_links': num_links,
                    'avg_cpu_util_per_node': f"{cpu_util:.4f}",
                    'avg_bw_util_per_link': f"{bw_util:.4f}",
                    'avg_mem_util_per_node': f"{mem_util:.4f}",
                    'total_hops': dep['total_hops'],
                    'avg_hops_per_dest': f"{avg_hops:.2f}",
                    'fully_deployed': dep['fully_deployed'],
                    'partial_deployed': dep['partial_deployed'],
                    'backup_used': dep['backup_used'],
                    'backup_levels': ','.join(dep['backup_levels']),
                    'failed_nodes': ','.join(map(str, dep['failed_nodes'])),
                    'failure_reasons': ','.join(dep['failure_reasons']),
                    'deployment_time': f"{dep['deployment_time']:.4f}"
                })

        print(f"✅ Deployment details exported to {filename}")

    def export_metrics_by_time_interval(self, filename: str = "metrics_by_time.csv", interval: int = 50):
        """
        [新增] 按时间间隔聚合数据 (用于绘制柱状图)
        聚合项：请求数量、接受率、CPU总消耗、带宽总消耗
        """
        if not self.deployment_history:
            return

        # 找出最大时间步
        max_t = max(d.get('time_step', 0) for d in self.deployment_history)

        # 准备分桶
        stats = defaultdict(lambda: {'total_req': 0, 'full_acc': 0, 'cpu': 0.0, 'bw': 0.0})

        for dep in self.deployment_history:
            t = dep.get('time_step', 0)
            # 计算桶索引：例如 1-50 -> 50, 51-100 -> 100
            if t == 0:
                bin_idx = interval
            else:
                bin_idx = ((t - 1) // interval + 1) * interval

            stats[bin_idx]['total_req'] += 1
            if dep['fully_deployed']:
                stats[bin_idx]['full_acc'] += 1

            # 使用 total_cpu_consumed 进行聚合 (总消耗)
            stats[bin_idx]['cpu'] += dep['total_cpu_consumed']
            stats[bin_idx]['bw'] += dep['total_bw_consumed']

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Time_Interval', 'Request_Count', 'Acceptance_Rate', 'Total_CPU_Consumed', 'Total_BW_Consumed'])

            for b in sorted(stats.keys()):
                d = stats[b]
                acc_rate = d['full_acc'] / d['total_req'] if d['total_req'] > 0 else 0
                writer.writerow([
                    b,
                    d['total_req'],
                    f"{acc_rate:.2%}",
                    f"{d['cpu']:.4f}",
                    f"{d['bw']:.4f}"
                ])

        print(f"✅ Aggregated time metrics exported to {filename}")

    def export_resource_utilization(self, filename: str = "resource_utilization.csv"):
        """导出资源利用率时序数据"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'request_id',
                'cpu_utilization', 'bw_utilization', 'mem_utilization',
                'num_used_nodes', 'num_used_links'
            ])

            writer.writeheader()

            for record in self.resource_utilization_history:
                writer.writerow({
                    'timestamp': record['timestamp'],
                    'request_id': record['request_id'],
                    'cpu_utilization': f"{record['cpu_utilization']:.4f}",
                    'bw_utilization': f"{record['bw_utilization']:.4f}",
                    'mem_utilization': f"{record['mem_utilization']:.4f}",
                    'num_used_nodes': record.get('num_used_nodes', 0),
                    'num_used_links': record.get('num_used_links', 0)
                })

        print(f"✅ Resource utilization exported to {filename}")

    def export_summary_statistics(self, filename: str = "summary_statistics.csv"):
        """导出汇总统计信息"""
        stats = self.compute_statistics()

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])

            # 接受率和阻塞率
            writer.writerow(['Total Requests', self.global_stats['total_requests']])
            writer.writerow(['Full Acceptance Rate', f"{stats['full_acceptance_rate']:.2%}"])
            writer.writerow(['Partial Acceptance Rate', f"{stats['partial_acceptance_rate']:.2%}"])
            writer.writerow(['Blocking Rate', f"{stats['blocking_rate']:.2%}"])
            writer.writerow([''])

            # 资源消耗
            writer.writerow(['Avg CPU per Deployment', f"{stats['avg_cpu_per_deployment']:.4f}"])
            writer.writerow(['Avg BW per Deployment', f"{stats['avg_bw_per_deployment']:.4f}"])
            writer.writerow(['Avg MEM per Deployment', f"{stats['avg_mem_per_deployment']:.4f}"])
            writer.writerow([''])

            # 资源利用率
            writer.writerow(['Avg CPU Utilization', f"{stats['avg_cpu_utilization']:.2%}"])
            writer.writerow(['Avg BW Utilization', f"{stats['avg_bw_utilization']:.2%}"])
            writer.writerow(['Avg MEM Utilization', f"{stats['avg_mem_utilization']:.2%}"])
            writer.writerow(['Std CPU Utilization', f"{stats['std_cpu_utilization']:.4f}"])
            writer.writerow(['Std BW Utilization', f"{stats['std_bw_utilization']:.4f}"])
            writer.writerow(['Std MEM Utilization', f"{stats['std_mem_utilization']:.4f}"])
            writer.writerow([''])

            # 其他指标
            writer.writerow(['Avg Hops per Destination', f"{stats['avg_hops']:.2f}"])
            writer.writerow(['Avg Deployment Time (s)', f"{stats['avg_deployment_time']:.4f}"])
            writer.writerow([''])

            # 失败节点TOP 10
            writer.writerow(['Top Failed Nodes', ''])
            for node, count in stats['top_failed_nodes']:
                writer.writerow([f'  Node {node}', count])
            writer.writerow([''])

            # 失败原因统计
            writer.writerow(['Failure Reasons', ''])
            for reason, count in stats['failure_reasons'].items():
                writer.writerow([f'  {reason}', count])
            writer.writerow([''])

            # 备份策略使用
            writer.writerow(['Backup Policy Usage', ''])
            for level, count in stats['backup_usage'].items():
                writer.writerow([f'  {level}', count])

        print(f"✅ Summary statistics exported to {filename}")

    def export_failed_nodes_analysis(self, filename: str = "failed_nodes_analysis.csv"):
        """导出失败节点的详细分析"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'node_id', 'failure_count', 'failure_rate',
                'main_failure_reasons'
            ])

            writer.writeheader()

            # 分析每个失败节点
            node_reasons = defaultdict(lambda: defaultdict(int))
            for dep in self.deployment_history:
                for node, reason in zip(dep['failed_nodes'], dep['failure_reasons']):
                    node_reasons[node][reason] += 1

            total_attempts = len(self.deployment_history)

            for node, count in self.global_stats["failed_nodes_count"].items():
                reasons = node_reasons[node]
                main_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)

                writer.writerow({
                    'node_id': node,
                    'failure_count': count,
                    'failure_rate': f"{count / total_attempts:.2%}",
                    'main_failure_reasons': '; '.join([
                        f"{r}({c})" for r, c in main_reasons[:3]
                    ])
                })

        print(f"✅ Failed nodes analysis exported to {filename}")

    def export_all(self, prefix: str = "vnf_metrics"):
        """一键导出所有CSV文件"""
        self.export_deployment_details(f"{prefix}_deployment_details.csv")
        self.export_resource_utilization(f"{prefix}_resource_utilization.csv")
        self.export_summary_statistics(f"{prefix}_summary_statistics.csv")
        self.export_failed_nodes_analysis(f"{prefix}_failed_nodes_analysis.csv")
        # [修改] 导出按时间间隔聚合的数据
        self.export_metrics_by_time_interval(f"{prefix}_metrics_by_time_interval.csv", interval=50)

        print(f"\n🎉 All metrics exported with prefix: {prefix}")

    # ================================================================
    # 📊 实时监控接口
    # ================================================================

    def get_realtime_stats(self) -> dict:
        """获取实时统计（用于训练过程中监控）"""
        if len(self.deployment_history) == 0:
            return {}

        recent_window = min(100, len(self.deployment_history))
        recent_deps = self.deployment_history[-recent_window:]

        fully_deployed = sum(1 for d in recent_deps if d['fully_deployed'])
        partially_deployed = sum(1 for d in recent_deps if d['partial_deployed'])

        return {
            "recent_full_acceptance": fully_deployed / recent_window,
            "recent_partial_acceptance": partially_deployed / recent_window,
            "recent_blocking": 1 - (fully_deployed + partially_deployed) / recent_window,
            "recent_avg_cpu_util": np.mean(self.global_stats["avg_cpu_utilization"][-recent_window:]),
            "recent_backup_usage": sum(1 for d in recent_deps if d['backup_used']) / recent_window
        }

    def print_summary(self):
        """打印汇总信息到控制台"""
        stats = self.compute_statistics()

        print("\n" + "=" * 60)
        print("📊 VNF DEPLOYMENT METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Requests: {self.global_stats['total_requests']}")
        print(f"Full Acceptance Rate: {stats['full_acceptance_rate']:.2%}")
        print(f"Partial Acceptance Rate: {stats['partial_acceptance_rate']:.2%}")
        print(f"Blocking Rate: {stats['blocking_rate']:.2%}")
        print("-" * 60)
        print(f"Avg CPU Utilization: {stats['avg_cpu_utilization']:.2%}")
        print(f"Avg BW Utilization: {stats['avg_bw_utilization']:.2%}")
        print(f"Avg MEM Utilization: {stats['avg_mem_utilization']:.2%}")
        print("-" * 60)
        print(f"Avg Hops per Destination: {stats['avg_hops']:.2f}")
        print(f"Avg Deployment Time: {stats['avg_deployment_time']:.4f}s")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # 简单的测试桩，验证代码是否可运行
    print("VNFMetricsLogger is ready.")