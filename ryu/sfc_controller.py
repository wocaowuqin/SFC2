# ==================== sfc_controller.py ====================
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
from collections import defaultdict
import time
import json


class SFCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SFCController, self).__init__(*args, **kwargs)

        # 网络拓扑
        self.topology = nx.DiGraph()
        self.datapaths = {}  # dpid -> datapath
        self.port_to_dpid = {}  # (dpid, port) -> neighbor_dpid

        # 网络状态监控
        self.link_stats = {}  # (src_dpid, dst_dpid) -> {bandwidth, delay, loss}
        self.node_stats = {}  # dpid -> {cpu_usage, memory, vnf_list}

        # SFC部署状态
        self.deployed_vnfs = {}  # vnf_id -> {node_id, type, resources}
        self.sfc_chains = {}  # sfc_id -> [vnf_id_list]
        self.flow_paths = {}  # flow_id -> [(dpid, in_port, out_port), ...]

        # MAC地址学习
        self.mac_to_port = {}  # dpid -> {mac: port}

        # 启动监控线程
        self.monitor_thread = hub.spawn(self._monitor_loop)

        self.logger.info("=" * 60)
        self.logger.info("SFC Controller initialized successfully")
        self.logger.info("Waiting for switches to connect...")
        self.logger.info("=" * 60)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """交换机连接时的初始化"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        self.datapaths[dpid] = datapath
        self.mac_to_port[dpid] = {}
        self.node_stats[dpid] = {
            'cpu_usage': 0.0,
            'memory': 0.0,
            'vnf_list': [],
            'available_cpu': 100.0,
            'available_memory': 8192.0
        }

        # 安装表项缺失时的默认流表：发送到控制器
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

        self.logger.info("+" * 60)
        self.logger.info(f"✓ Switch {dpid} connected and configured")
        self.logger.info(f"  Total switches: {len(self.datapaths)}")
        self.logger.info("+" * 60)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """监控交换机状态变化"""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"Register datapath: {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.warning(f"Unregister datapath: {datapath.id}")
                del self.datapaths[datapath.id]

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """添加流表项"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        datapath.send_msg(mod)

    def delete_flow(self, datapath, match):
        """删除流表项"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """处理PacketIn消息（未匹配流表的数据包）"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return  # 忽略LLDP包

        dst = eth.dst
        src = eth.src

        # MAC地址学习
        self.mac_to_port[dpid][src] = in_port

        # 查找目标端口
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # 如果不是泛洪，安装流表
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions, idle_timeout=30)

        # 发送PacketOut
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """交换机加入拓扑"""
        self.logger.info(f"Switch entered topology: {ev.switch.dp.id}")
        hub.sleep(1)  # 等待拓扑稳定
        self._update_topology()

    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        """链路添加"""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        src_port = link.src.port_no
        dst_port = link.dst.port_no

        self.topology.add_edge(src_dpid, dst_dpid,
                               port=src_port,
                               weight=1,
                               bandwidth=1000.0,  # Mbps
                               delay=1.0,  # ms
                               loss=0.0)

        self.port_to_dpid[(src_dpid, src_port)] = dst_dpid

        self.logger.info(f"Link added: s{src_dpid}:{src_port} -> s{dst_dpid}:{dst_port}")

    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        """链路删除"""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        
        if self.topology.has_edge(src_dpid, dst_dpid):
            self.topology.remove_edge(src_dpid, dst_dpid)
            self.logger.warning(f"Link deleted: s{src_dpid} -> s{dst_dpid}")

    def _update_topology(self):
        """更新拓扑信息"""
        try:
            switches = get_switch(self, None)
            links = get_link(self, None)

            # 清除旧拓扑
            self.topology.clear()

            # 添加交换机节点
            for switch in switches:
                self.topology.add_node(switch.dp.id)

            # 添加链路
            for link in links:
                src = link.src.dpid
                dst = link.dst.dpid
                self.topology.add_edge(src, dst,
                                       port=link.src.port_no,
                                       weight=1,
                                       bandwidth=1000.0,
                                       delay=1.0,
                                       loss=0.0)

            self.logger.info(f"Topology updated: {len(switches)} switches, {len(links)} links")
            
            # 打印拓扑信息
            if len(switches) > 0:
                self.logger.info(f"Switches: {[s.dp.id for s in switches]}")
                self.logger.info(f"Links: {[(l.src.dpid, l.dst.dpid) for l in links]}")
                
        except Exception as e:
            self.logger.error(f"Error updating topology: {e}")

    def _monitor_loop(self):
        """周期性监控网络状态"""
        while True:
            if len(self.datapaths) > 0:
                self._request_stats()
            hub.sleep(10)  # 每10秒监控一次

    def _request_stats(self):
        """请求交换机统计信息"""
        for dpid, datapath in list(self.datapaths.items()):
            try:
                parser = datapath.ofproto_parser
                ofproto = datapath.ofproto

                # 请求端口统计
                req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
                datapath.send_msg(req)

                # 请求流表统计
                req = parser.OFPFlowStatsRequest(datapath)
                datapath.send_msg(req)
            except Exception as e:
                self.logger.error(f"Error requesting stats from switch {dpid}: {e}")

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """处理端口统计回复"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            port_no = stat.port_no
            if port_no < 65280:  # 排除虚拟端口
                # 计算带宽使用率等
                rx_bytes = stat.rx_bytes
                tx_bytes = stat.tx_bytes
                # 可以存储和分析这些数据

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """处理流表统计回复"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        flows = []
        for stat in body:
            flows.append({
                'priority': stat.priority,
                'match': stat.match,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count
            })
        
        # 可以分析流表使用情况

    def compute_path(self, src_dpid, dst_dpid, metric='shortest'):
        """
        计算路径
        Args:
            src_dpid: 源交换机
            dst_dpid: 目标交换机
            metric: 路径选择策略 ('shortest', 'min_delay', 'max_bandwidth')
        """
        try:
            if metric == 'shortest':
                path = nx.shortest_path(self.topology, src_dpid, dst_dpid)
            elif metric == 'min_delay':
                path = nx.shortest_path(self.topology, src_dpid, dst_dpid, weight='delay')
            else:
                path = nx.shortest_path(self.topology, src_dpid, dst_dpid)

            return path
        except nx.NetworkXNoPath:
            self.logger.error(f"No path found between {src_dpid} and {dst_dpid}")
            return None
        except Exception as e:
            self.logger.error(f"Error computing path: {e}")
            return None

    def install_sfc_path(self, path, flow_match, priority=10):
        """
        沿路径安装流表
        Args:
            path: 交换机路径 [dpid1, dpid2, ...]
            flow_match: 流匹配规则字典
            priority: 流表优先级
        """
        try:
            for i in range(len(path) - 1):
                current_dpid = path[i]
                next_dpid = path[i + 1]

                if current_dpid not in self.datapaths:
                    self.logger.error(f"Datapath {current_dpid} not found")
                    continue

                datapath = self.datapaths[current_dpid]
                parser = datapath.ofproto_parser

                # 查找输出端口
                out_port = self.topology[current_dpid][next_dpid]['port']

                # 构建匹配和动作
                match = parser.OFPMatch(**flow_match)
                actions = [parser.OFPActionOutput(out_port)]

                self.add_flow(datapath, priority, match, actions)

            self.logger.info(f"SFC path installed: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error installing SFC path: {e}")
            return False

    def get_network_state(self):
        """获取当前网络状态（供DRL Agent使用）"""
        state = {
            'topology': {
                'nodes': list(self.topology.nodes()),
                'edges': list(self.topology.edges())
            },
            'node_stats': self.node_stats,
            'link_stats': self.link_stats,
            'deployed_vnfs': self.deployed_vnfs,
            'timestamp': time.time()
        }
        return state

    def deploy_vnf(self, vnf_id, vnf_type, target_node, resources):
        """
        部署VNF到指定节点
        Args:
            vnf_id: VNF标识
            vnf_type: VNF类型 (firewall, ids, load_balancer, etc.)
            target_node: 目标节点(dpid)
            resources: 资源需求 {cpu, memory}
        Returns:
            success: 是否成功
            message: 错误信息
        """
        # 检查资源
        if target_node not in self.node_stats:
            return False, f"Node {target_node} not found"

        node = self.node_stats[target_node]
        if node['available_cpu'] < resources.get('cpu', 0):
            return False, f"Insufficient CPU on node {target_node}"
        if node['available_memory'] < resources.get('memory', 0):
            return False, f"Insufficient memory on node {target_node}"

        # 更新资源
        node['available_cpu'] -= resources.get('cpu', 0)
        node['available_memory'] -= resources.get('memory', 0)
        node['vnf_list'].append(vnf_id)

        # 记录部署
        self.deployed_vnfs[vnf_id] = {
            'node_id': target_node,
            'type': vnf_type,
            'resources': resources,
            'deployed_at': time.time()
        }

        self.logger.info(f"✓ VNF {vnf_id} ({vnf_type}) deployed to node {target_node}")
        return True, "Success"

    def export_topology_to_xml(self, filename='topology_export.xml'):
        """导出拓扑为XML格式（兼容训练框架）"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.Element("network")
            topology = ET.SubElement(root, "topology")
            
            # 添加节点
            for node in self.topology.nodes():
                node_elem = ET.SubElement(topology, "node")
                node_elem.set("id", str(node))
                stats = self.node_stats.get(node, {})
                node_elem.set("cpu_total", str(stats.get('available_cpu', 100.0)))
            
            # 添加链路
            for src, dst in self.topology.edges():
                link_elem = ET.SubElement(topology, "link")
                from_elem = ET.SubElement(link_elem, "from")
                from_elem.set("node", str(src))
                to_elem = ET.SubElement(link_elem, "to")
                to_elem.set("node", str(dst))
                
                edge_data = self.topology[src][dst]
                link_elem.set("bw", str(edge_data.get('bandwidth', 1000.0)))
                link_elem.set("delay", str(edge_data.get('delay', 1.0)))
                link_elem.set("loss", str(edge_data.get('loss', 0.0)))
            
            tree = ET.ElementTree(root)
            tree.write(filename)
            self.logger.info(f"Topology exported to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting topology: {e}")
            return False
