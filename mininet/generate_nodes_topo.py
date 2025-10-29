# -*- coding: utf-8 -*-
# @File    : generate_nodes_topo_py2.py
# @Date    : 2021-12-09
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @Modified: converted for Python2 by ChatGPT

import os
import random
import time
import json
import threading
import xml.etree.ElementTree as ET
import sys

# ensure system site-packages available (sometimes needed)
sys.path.append("/usr/lib/python2.7/dist-packages")

import networkx
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.util import dumpNodeConnections

random.seed(2020)


def generate_port(node_idx1, node_idx2):
    if (node_idx2 > 9) and (node_idx1 > 9):
        port = str(node_idx1) + "0" + str(node_idx2)
    else:
        port = str(node_idx1) + "00" + str(node_idx2)  # test
    return int(port)


def generate_switch_port(graph):
    switch_port_dict = {}
    for node in graph.nodes():
        # degree for networkx 1.x/2.x: graph.degree[node] or graph.degree(node)
        try:
            deg = graph.degree[node]
        except Exception:
            deg = graph.degree(node)
        switch_port_dict.setdefault(node, list(range(deg)))
    return switch_port_dict


def parse_xml_topology(topology_path):
    """
        parse topology from topology.xml
    :return: topology graph, networkx.Graph()
             nodes_num,  int
             edges_num, int
    """
    tree = ET.parse(topology_path)
    root = tree.getroot()
    topo_element = root.find("topology")
    graph = networkx.Graph()
    for child in topo_element.iter():
        # parse nodes
        if child.tag == 'node':
            node_id = int(child.get('id'))
            graph.add_node(node_id)
        # parse link
        elif child.tag == 'link':
            from_node = int(child.find('from').get('node'))
            to_node = int(child.find('to').get('node'))
            graph.add_edge(from_node, to_node)

    # nodes_num, edges_num
    try:
        nodes_num = len(graph.nodes())
        edges_num = len(graph.edges())
    except Exception:
        nodes_num = len(graph.nodes)
        edges_num = len(graph.edges)

    print('nodes: ', nodes_num, '\n', graph.nodes(), '\n', 'edges: ', edges_num, '\n', graph.edges())
    return graph, nodes_num, edges_num


def create_topo_links_info_xml(path, links_info):
    """
        create XML for links_info
    """
    root = ET.Element('links_info')

    for link, info in links_info.items():
        child = ET.SubElement(root, 'links')
        child.text = str(link)

        sub_child1 = ET.SubElement(child, 'ports')
        sub_child1.text = str((info['port1'], info['port2']))

        sub_child2 = ET.SubElement(child, 'bw')
        sub_child2.text = str(info['bw'])

        sub_child3 = ET.SubElement(child, 'delay')
        sub_child3.text = str(info['delay'])

        sub_child4 = ET.SubElement(child, 'loss')
        sub_child4.text = str(info['loss'])

    tree = ET.ElementTree(root)
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print('saved links info xml to %s' % path)


def get_mininet_device(net, idx, device='h'):
    """
        Get mininet device instances for indices in idx
        :param net: Mininet instance
        :param idx: list of indices
        :param device: 'h' for hosts or 's' for switches
    :return: dict mapping index -> device instance
    """
    d = {}
    for i in idx:
        d.setdefault(i, net.get('{}{}'.format(device, i)))
    return d


def run_corresponding_sh_script(devices, label_path):
    """
        For each device run the corresponding shell script.
        label_path example: './24nodes/TM-{}/{}/{}_'
    """
    # label_path expected to contain formatting placeholders at end
    for i, d in devices.items():
        if i < 9:
            idx_str = '0{}'.format(i)
        else:
            idx_str = str(i)
        p = label_path + '{}.sh'
        script_path = p.format(idx_str)
        _cmd = 'bash {}'.format(script_path)
        d.cmd(_cmd)
    print("---> complete run {}".format(label_path))


def run_ip_add_default(hosts):
    """
        Run ip route add default via 10.0.0.x for each host
    """
    _cmd = 'ip route add default via 10.0.0.'
    for i, h in hosts.items():
        print(_cmd + str(i))
        h.cmd(_cmd + str(i))
    print("---> run ip add default complete")


def _test_cmd(devices, my_cmd):
    for i, d in devices.items():
        d.cmd(my_cmd)
        print('exec {} zzz{}'.format(my_cmd, i))


def run_iperf(path, host):
    _cmd = 'bash ' + path + ' &'
    host.cmd(_cmd)


def all_host_run_iperf(hosts, path, finish_file):
    """
        path = './iperfTM/'
    """
    if not os.path.isdir(path):
        print("iperf path does not exist:", path)
        return

    idxs = len(os.listdir(path))
    # path like ./iperfTM/TM-<idx>
    basepath = path + '/TM-'
    for idx in range(idxs):
        script_path = basepath + str(idx)
        for i, h in hosts.items():
            servers_cmd = script_path + '/Servers/server_' + str(i) + '.sh'
            _cmd = 'bash ' + servers_cmd
            print(_cmd)
            h.cmd(_cmd)

        for i, h in hosts.items():
            clients_cmd = script_path + '/Clients/client_' + str(i) + '.sh'
            _cmd = 'bash ' + clients_cmd
            print(_cmd + " " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            h.cmd(_cmd)

        time.sleep(300)

    write_iperf_time(finish_file)


def write_pinall_time(finish_file):
    with open(finish_file, "w+") as f:
        _content = {
            "ping_all_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "start_save_flag": True,
            "finish_flag": False
        }
        json.dump(_content, f)


def write_iperf_time(finish_file):
    try:
        with open(finish_file, "r+") as f:
            _read = json.load(f)
    except Exception:
        _read = {}

    _content = {
        "iperf_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "finish_flag": True,
    }
    _read.update(_content)

    with open(finish_file, "w+") as f:
        json.dump(_read, f)


def remove_finish_file(finish_file):
    try:
        if os.path.exists(finish_file):
            os.remove(finish_file)
    except Exception:
        pass


def net_h1_ping_others(net):
    hosts = net.hosts
    # ping from hosts[0] to others
    for h in hosts[1:]:
        net.ping((hosts[0], h))


class GEANT23nodesTopo(Topo):
    def __init__(self, graph):
        Topo.__init__(self)
        self.node_idx = list(graph.nodes())
        self.edges_pairs = list(graph.edges())
        self.bw1 = 100  # Gbps -> M
        self.bw2 = 25  # Gbps -> M
        self.bw3 = 1.15  # Mbps
        self.bw4 = 100  # host -- switch
        self.delay = 20
        self.loss = 10

        self.host_port = 9
        self.snooper_port = 10

        self.bw1_links = [(12, 22), (12, 10), (12, 2), (13, 17), (4, 2), (4, 16), (1, 3), (1, 7), (1, 16), (3, 10),
                          (3, 21), (10, 16), (10, 17), (7, 17), (7, 2), (7, 21), (16, 9), (20, 17)]
        self.bw2_links = [(13, 19), (13, 2), (19, 7), (23, 17), (23, 2), (8, 5), (8, 9), (18, 2), (18, 21), (5, 16),
                          (3, 11), (10, 11), (22, 20), (20, 15), (9, 15)]
        self.bw3_links = [(13, 14), (19, 6), (3, 14), (7, 6)]

        def _return_bw(link):
            if link in self.bw1_links:
                return self.bw1
            elif link in self.bw2_links:
                return self.bw2
            elif link in self.bw3_links:
                return self.bw3
            else:
                raise ValueError

        # add switches
        switches = {}
        for s in self.node_idx:
            switches.setdefault(s, self.addSwitch('s{0}'.format(s)))
            print('添加交换机:', s)

        switch_port_dict = generate_switch_port(graph)
        links_info = {}
        # add links
        for l in self.edges_pairs:
            port1 = switch_port_dict[l[0]].pop(0) + 1
            port2 = switch_port_dict[l[1]].pop(0) + 1
            bw = _return_bw(l)

            _d = str(random.randint(0, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)

            self.addLink(switches[l[0]], switches[l[1]], port1=port1, port2=port2,
                         bw=bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": bw, "delay": _d, "loss": _l})

        # save links info xml
        create_topo_links_info_xml(links_info_xml_path, links_info)

        # add hosts
        for i in self.node_idx:
            mac = '00.00.00.00.00.0{0}'.format(i)
            ip = '10.0.0.{0}'.format(i)
            _h = self.addHost('h{0}'.format(i), ip=ip, mac=mac)
            self.addLink(_h, switches[i], port1=0, port2=self.host_port,
                         bw=self.bw4)


class Nodes14Topo(Topo):
    def __init__(self, graph):
        Topo.__init__(self)
        self.node_idx = list(graph.nodes())
        self.edges_pairs = list(graph.edges())

        self.random_bw = 30  # Gbps -> M * 10
        self.bw4 = 50  # host -- switch

        self.delay = 20  # ms
        self.loss = 10  # %

        self.host_port = 9
        self.snooper_port = 10

        # add switches
        switches = {}
        for s in self.node_idx:
            switches.setdefault(s, self.addSwitch('s{0}'.format(s)))
            print('添加交换机:', s)

        switch_port_dict = generate_switch_port(graph)
        links_info = {}
        # add links
        for l in self.edges_pairs:
            port1 = switch_port_dict[l[0]].pop(0) + 1
            port2 = switch_port_dict[l[1]].pop(0) + 1

            _bw = random.randint(5, self.random_bw)
            _d = str(random.randint(1, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)

            # 构建链路参数，仅在丢包率>0时添加loss参数
            link_kwargs = {
                'port1': port1,
                'port2': port2,
                'bw': _bw,
                'delay': _d
            }
            # 只有当丢包率大于0时才添加loss参数
            if _l > 0:
                link_kwargs['loss'] = _l

            self.addLink(switches[l[0]], switches[l[1]], **link_kwargs)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": _bw, "delay": _d, "loss": _l})

        create_topo_links_info_xml(links_info_xml_path, links_info)

        # add hosts
        for i in self.node_idx:
            mac = '00.00.00.00.00.0{0}'.format(i)
            ip = '10.0.0.{0}'.format(i)
            _h = self.addHost('h{0}'.format(i), ip=ip, mac=mac)
            self.addLink(_h, switches[i], port1=0, port2=self.host_port,
                         bw=self.bw4)

def main(graph, topo, finish_file):
    print('===Remove old finish file')
    remove_finish_file(finish_file)

    net = Mininet(topo=topo, link=TCLink, controller=RemoteController, waitConnected=True, build=False)
    c0 = net.addController('c0', ip='127.0.0.1', port=6633)

    net.build()
    net.start()

    print("get hosts device list")
    hosts = get_mininet_device(net, graph.nodes(), device='h')

    print("===Dumping host connections")
    dumpNodeConnections(net.hosts)
    print('===Wait ryu init')
    time.sleep(40)
    # add gateway ip if needed
    # run_ip_add_default(hosts)

    net_h1_ping_others(net)
    write_pinall_time(finish_file)

    print('===Run iperf scripts')
    t = threading.Thread(target=all_host_run_iperf, args=(hosts, iperf_path, finish_file), name='iperf')
    t.daemon = True
    t.start()

    CLI(net)
    net.stop()


if __name__ == '__main__':
    xml_topology_path = './topologies/topology2.xml'
    links_info_xml_path = './links_info/links_info.xml'
    iperf_path = "./iperfTM"
    iperf_interval = 0
    finish_file = './finish_time.json'

    graph, nodes_num, edges_num = parse_xml_topology(xml_topology_path)
    topo = Nodes14Topo(graph)

    setLogLevel('info')
    main(graph, topo, finish_file)
