from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
from collections import defaultdict
import time


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

        self.logger.info("SFC Controller initialized")

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

        self.logger.info(f"Switch {dpid} connected")

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
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """交换机加入拓扑"""
        self.logger.info(f"Switch entered: {ev.switch.dp.id}")
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

        self.logger.info(f"Link added: {src_dpid}:{src_port} -> {dst_dpid}:{dst_port}")

    def _update_topology(self):
        """更新拓扑信息"""
        switches = get_switch(self, None)
        links = get_link(self, None)

        self.topology.clear()

        for switch in switches:
            self.topology.add_node(switch.dp.id)

        for link in links:
            src = link.src.dpid
            dst = link.dst.dpid
            self.topology.add_edge(src, dst,
                                   port=link.src.port_no,
                                   weight=1)

    def _monitor_loop(self):
        """周期性监控网络状态"""
        while True:
            self._request_stats()
            hub.sleep(5)  # 每5秒监控一次

    def _request_stats(self):
        """请求交换机统计信息"""
        for dpid, datapath in self.datapaths.items():
            parser = datapath.ofproto_parser

            # 请求端口统计
            req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
            datapath.send_msg(req)

            # 请求流表统计
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """处理端口统计回复"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            port_no = stat.port_no
            # 可以计算带宽使用率、丢包率等
            # 这里简化处理
            pass

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

    def install_sfc_path(self, path, flow_match, priority=10):
        """
        沿路径安装流表
        Args:
            path: 交换机路径 [dpid1, dpid2, ...]
            flow_match: 流匹配规则 (src_ip, dst_ip, protocol, ...)
            priority: 流表优先级
        """
        for i in range(len(path) - 1):
            current_dpid = path[i]
            next_dpid = path[i + 1]

            datapath = self.datapaths[current_dpid]
            parser = datapath.ofproto_parser

            # 查找输出端口
            out_port = self.topology[current_dpid][next_dpid]['port']

            # 构建匹配和动作
            match = parser.OFPMatch(**flow_match)
            actions = [parser.OFPActionOutput(out_port)]

            self.add_flow(datapath, priority, match, actions)

        self.logger.info(f"SFC path installed: {path}")

    def get_network_state(self):
        """获取当前网络状态（供HRL Agent使用）"""
        state = {
            'topology': nx.node_link_data(self.topology),
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
        if node['available_cpu'] < resources['cpu']:
            return False, f"Insufficient CPU on node {target_node}"
        if node['available_memory'] < resources['memory']:
            return False, f"Insufficient memory on node {target_node}"

        # 更新资源
        node['available_cpu'] -= resources['cpu']
        node['available_memory'] -= resources['memory']
        node['vnf_list'].append(vnf_id)

        # 记录部署
        self.deployed_vnfs[vnf_id] = {
            'node_id': target_node,
            'type': vnf_type,
            'resources': resources,
            'deployed_at': time.time()
        }

        self.logger.info(f"VNF {vnf_id} deployed to node {target_node}")
        return True, "Success"