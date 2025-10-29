# -*- coding: utf-8 -*-
# (Python 2 不需要上面这行，但加上通常无害)

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time


class SFCTopology(Topo):
    """
    SFC网络拓扑 (Python 2 版本)
    创建一个适合SFC部署的数据中心拓扑
    """

    def __init__(self, num_switches=6, num_hosts_per_switch=2):
        # Python 2 中 super 需要明确指定类名和 self
        super(SFCTopology, self).__init__()

        switches = []
        hosts = []

        # 创建交换机
        for i in range(1, num_switches + 1):
            # 使用 % 格式化字符串
            switch = self.addSwitch('s%d' % i, protocols='OpenFlow13')
            switches.append(switch)

        # 创建主机（模拟物理节点）
        host_id = 1
        for switch in switches:
            for j in range(num_hosts_per_switch):
                # 使用 % 格式化字符串
                host_name = 'h%d' % host_id
                ip_addr = '10.0.0.%d' % host_id
                mac_addr = '00:00:00:00:00:%02x' % host_id # %02x 用于格式化十六进制

                host = self.addHost(host_name, ip=ip_addr, mac=mac_addr)
                self.addLink(host, switch,
                             bw=1000,  # 1 Gbps
                             delay='1ms',
                             loss=0)
                hosts.append(host)
                host_id += 1

        # 创建交换机之间的链路（Fat-Tree 拓扑的简化版本）
        # 核心层
        core_switches = switches[:2]  # s1, s2
        # 汇聚层
        agg_switches = switches[2:4]  # s3, s4
        # 接入层
        edge_switches = switches[4:]  # s5, s6

        # 核心层到汇聚层的连接
        for core in core_switches:
            for agg in agg_switches:
                self.addLink(core, agg,
                             bw=1000,
                             delay='2ms',
                             loss=0)

        # 汇聚层到接入层的连接
        for i, agg in enumerate(agg_switches):
            # 每个汇聚交换机连接到对应的接入交换机
            if i < len(edge_switches):
                self.addLink(agg, edge_switches[i],
                             bw=1000,
                             delay='1ms',
                             loss=0)

        # 接入层之间的连接（用于冗余）
        if len(edge_switches) >= 2:
            self.addLink(edge_switches[0], edge_switches[1],
                         bw=500,
                         delay='3ms',
                         loss=0)


def setup_network():
    """启动Mininet网络 (Python 2 版本)"""
    setLogLevel('info')

    # 创建拓扑
    topo = SFCTopology(num_switches=6, num_hosts_per_switch=2)

    # 创建网络，连接到Ryu控制器
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name,
            ip='127.0.0.1',  # Ryu控制器地址
            port=6653  # OpenFlow端口
        ),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    # Python 2 print 语句
    info('*** Starting network\n')
    net.start()

    # Python 2 print 语句
    info('*** Waiting for controller connection...\n')
    time.sleep(3)

    # Python 2 print 语句
    info('*** Testing connectivity\n')
    net.pingAll()

    # Python 2 print 语句
    info('*** Network topology:\n')
    for switch in net.switches:
        # 使用 % 格式化字符串
        info('Switch %s: \n' % switch.name)
        info('  Ports: %s\n' % switch.ports) # switch.ports 可能需要转换成字符串

    for host in net.hosts:
        # 使用 % 格式化字符串
        info('Host %s: IP=%s, MAC=%s\n' % (host.name, host.IP(), host.MAC()))

    return net


def configure_hosts(net):
    """配置主机，模拟VNF节点 (Python 2 版本)"""
    # Python 2 print 语句
    info('*** Configuring hosts as VNF nodes\n')

    for host in net.hosts:
        # 设置每个主机的处理能力（模拟CPU核心数）
        host.cpu_cores = 4
        host.memory_mb = 4096
        host.deployed_vnfs = []

        # 启动简单的HTTP服务器来模拟VNF
        # Python 2 中模块名为 SimpleHTTPServer
        # host.cmd('python -m SimpleHTTPServer 80 &')

        # 使用 % 格式化字符串
        info('Host %s: %d cores, %dMB RAM\n' % (host.name, host.cpu_cores, host.memory_mb))


def generate_traffic(net, src_host, dst_host, duration=10):
    """生成测试流量 (Python 2 版本)"""
    # Python 2 print 语句
    info('*** Generating traffic from %s to %s\n' % (src_host.name, dst_host.name))

    # 使用iperf生成TCP流量
    dst_host.cmd('iperf -s &')
    time.sleep(1)

    # 使用 % 格式化字符串
    result = src_host.cmd('iperf -c %s -t %d' % (dst_host.IP(), duration))
    info(result + '\n') # 确保换行

    dst_host.cmd('kill %iperf')


def cleanup_network(net):
    """清理网络 (Python 2 版本)"""
    # Python 2 print 语句
    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    # 启动网络
    net = setup_network()

    # 配置主机
    configure_hosts(net)

    try:
        # 进入CLI，允许手动测试
        # Python 2 print 语句
        info('*** Running CLI (type "exit" to quit)\n')
        info('*** Ryu controller should be running at 127.0.0.1:6653\n')
        info('*** HRL agent can now interact via REST API at http://localhost:8080/sfc/\n')
        CLI(net)
    finally:
        # 确保无论如何都执行清理
        cleanup_network(net)