from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time


class SFCTopology(Topo):
    """
    SFC网络拓扑
    创建一个适合SFC部署的数据中心拓扑
    """

    def __init__(self, num_switches=6, num_hosts_per_switch=2):
        Topo.__init__(self)

        switches = []
        hosts = []

        # 创建交换机
        for i in range(1, num_switches + 1):
            switch = self.addSwitch(f's{i}', protocols='OpenFlow13')
            switches.append(switch)

        # 创建主机（模拟物理节点）
        host_id = 1
        for switch in switches:
            for j in range(num_hosts_per_switch):
                host = self.addHost(f'h{host_id}',
                                    ip=f'10.0.0.{host_id}',
                                    mac=f'00:00:00:00:00:{host_id:02x}')
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
    """启动Mininet网络"""
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

    info('*** Starting network\n')
    net.start()

    # 等待控制器连接
    info('*** Waiting for controller connection...\n')
    time.sleep(3)

    # 测试连通性
    info('*** Testing connectivity\n')
    net.pingAll()

    # 打印拓扑信息
    info('*** Network topology:\n')
    for switch in net.switches:
        info(f'Switch {switch.name}: ')
        info(f'  Ports: {switch.ports}\n')

    for host in net.hosts:
        info(f'Host {host.name}: IP={host.IP()}, MAC={host.MAC()}\n')

    return net


def configure_hosts(net):
    """配置主机，模拟VNF节点"""
    info('*** Configuring hosts as VNF nodes\n')

    for host in net.hosts:
        # 设置每个主机的处理能力（模拟CPU核心数）
        host.cpu_cores = 4
        host.memory_mb = 4096
        host.deployed_vnfs = []

        # 启动简单的HTTP服务器来模拟VNF
        # host.cmd('python -m SimpleHTTPServer 80 &')

        info(f'Host {host.name}: {host.cpu_cores} cores, {host.memory_mb}MB RAM\n')


def generate_traffic(net, src_host, dst_host, duration=10):
    """生成测试流量"""
    info(f'*** Generating traffic from {src_host.name} to {dst_host.name}\n')

    # 使用iperf生成TCP流量
    dst_host.cmd('iperf -s &')
    time.sleep(1)

    result = src_host.cmd(f'iperf -c {dst_host.IP()} -t {duration}')
    info(result)

    dst_host.cmd('kill %iperf')


def cleanup_network(net):
    """清理网络"""
    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    # 启动网络
    net = setup_network()

    # 配置主机
    configure_hosts(net)

    try:
        # 进入CLI，允许手动测试
        info('*** Running CLI (type "exit" to quit)\n')
        info('*** Ryu controller should be running at 127.0.0.1:6653\n')
        info('*** HRL agent can now interact via REST API at http://localhost:8080/sfc/\n')
        CLI(net)
    finally:
        cleanup_network(net)