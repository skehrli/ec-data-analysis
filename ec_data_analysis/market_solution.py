#!/usr/bin/env python3

from .battery import Battery
from .constants import SOURCE, NetworkAlloc, TARGET, BATTERY, UNBOUNDED
import pandas as pd
import networkx as nx
from typing import Optional
from typing import List
import matplotlib.pyplot as plt
from functools import cached_property


class MarketSolution:
    data_vec: pd.Series
    # the 'fair' network for market allocation
    N_fair: nx.DiGraph
    # depends only on data_vec
    tradingVolume: float
    sellMap: NetworkAlloc

    # depends also on battery capacity/status
    chargeAmount: float
    chargeMap: Optional[NetworkAlloc]
    dischargeAmount: float
    dischargeMap: Optional[NetworkAlloc]

    def __init__(self, data_vec: pd.Series) -> None:
        """
        data_vec is a vector, where data_vec.iloc[i] is the amount of w/h member i
        is selling if the value is non-negative or the amount member i is buying else.
        """
        self.chargeAmount = 0
        self.chargeMap = None
        self.dischargeAmount = 0
        self.dischargeMap = None
        self.data_vec = data_vec
        self.N_fair = self._construct_fair_network(data_vec)
        self.tradingVolume, self.sellMap = nx.maximum_flow(self.N_fair, SOURCE, TARGET)

    def computeWithBattery(self, battery: Battery) -> None:
        supply: float = self.getSupply
        demand: float = self.getDemand
        N: nx.DiGraph
        if supply >= demand:
            N = self._construct_charging_network(battery)
            self.chargeAmount, self.chargeMap = nx.maximum_flow(N, SOURCE, TARGET)
            battery.charge(self.chargeAmount)
        else:
            N = self._construct_discharging_network(battery)
            self.dischargeAmount, self.dischargeMap = nx.maximum_flow(N, SOURCE, TARGET)
            battery.discharge(self.dischargeAmount)

    @cached_property
    def getSupply(self) -> float:
        return self.data_vec[self.data_vec >= 0].sum()

    @cached_property
    def getDemand(self) -> float:
        return -self.data_vec[self.data_vec < 0].sum()

    def getQtySoldForMember(self, member: int) -> float:
        node: str = self._get_node(member)
        return self.sellMap[SOURCE].get(node, 0)

    def getQtyPurchasedForMember(self, member: int) -> float:
        node: str = self._get_node(member)
        return self.sellMap[node].get(TARGET, 0)

    def getQtyChargedForMember(self, member: int) -> float:
        node: str = self._get_node(member)
        if self.chargeMap is not None:
            return self.chargeMap.get(node, {}).get(BATTERY, 0)
        else:
            return 0

    def getQtyDischargedForMember(self, member: int) -> float:
        node: str = self._get_node(member)
        if self.dischargeMap is not None:
            return self.dischargeMap[BATTERY].get(node, 0)
        else:
            return 0

    def plot_flow_graph(self) -> None:
        """
        Plots the flow graph in layers based on the self.sellMap and the maximum flow value.

        The graph is plotted such that SOURCE is on the left, TARGET is on the right,
        and other nodes are layered according to their connections.

        Returns:
        None
        """
        flow_graph = nx.DiGraph()

        # Add edges with flow > 0 to the flow graph
        for u in self.sellMap:
            for v, flow in self.sellMap[u].items():
                if flow > 0:
                    flow_graph.add_edge(u, v, weight=flow)

        # Assign layer/subset to each node
        for node in flow_graph.nodes:
            flow_graph.nodes[node]["layer"] = self._get_layer(node, flow_graph)

        # Create a layered layout (hierarchical layout from left to right)
        pos = nx.multipartite_layout(flow_graph, subset_key="layer")

        plt.figure(figsize=(10, 6))

        # Draw nodes
        nx.draw_networkx_nodes(flow_graph, pos, node_size=700)

        # Draw edges with widths proportional to the flow value
        edge_weights = [flow_graph[u][v]["weight"] for u, v in flow_graph.edges()]
        nx.draw_networkx_edges(flow_graph, pos, width=edge_weights)

        # Draw node labels
        nx.draw_networkx_labels(flow_graph, pos, font_size=14)

        # Draw edge labels (flow values rounded to two decimal points)
        edge_labels = {
            (u, v): f'{flow_graph[u][v]["weight"]:.2f}' for u, v in flow_graph.edges()
        }
        nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels)

        # Show the plot
        plt.title(f"Flow Network with Maximum Flow Value: {self.tradingVolume:.2f}")
        plt.axis("off")
        plt.show()

    def _get_node(self, n: int) -> str:
        """
        Defines mapping from index in list to node name. Currently just casts to string.
        Checks the invariant that no node has the same name as SOURCE or TARGET

        Parameters:
        n (int): An index in list

        Returns:
        str: The corresponding node name. Currently just the string representation of that integer.

        Raises:
        AssertionError: If the converted string is equal to SOURCE or TARGET.
        """
        node = str(n)
        assert node != SOURCE and node != TARGET and node != BATTERY
        return node

    def _construct_charging_network(self, battery: Battery) -> nx.DiGraph:
        N: nx.DiGraph = nx.DiGraph()
        N.add_node(BATTERY)
        N.add_node(SOURCE)
        N.add_node(TARGET)
        N.add_edge(BATTERY, TARGET, capacity=battery.chargeAmount())
        for i, val in enumerate(self.data_vec):
            if val >= 0:
                node = self._get_node(i)
                N.add_node(node)
                N.add_edge(node, BATTERY, capacity=UNBOUNDED)
                cap: float = val - self.sellMap[SOURCE].get(node, 0)
                N.add_edge(SOURCE, node, capacity=cap)
        return N

    def _construct_discharging_network(self, battery: Battery) -> nx.DiGraph:
        N: nx.DiGraph = nx.DiGraph()
        N.add_node(BATTERY)
        N.add_node(SOURCE)
        N.add_node(TARGET)
        N.add_edge(SOURCE, BATTERY, capacity=battery.dischargeAmount())
        for i, val in enumerate(self.data_vec):
            if val < 0:
                node = self._get_node(i)
                N.add_node(node)
                N.add_edge(BATTERY, node, capacity=UNBOUNDED)
                cap: float = -val - self.sellMap[node].get(TARGET, 0)
                N.add_edge(node, TARGET, capacity=cap)
        return N

    def _construct_fair_network(self, vals: pd.Series) -> nx.DiGraph:
        """
        Constructs a directed graph (`nx.DiGraph`) to run a maximum flow algorithm from SOURCE to TARGET.

        The network is built based on the following rules:
        - A non-negative value `v` in `vals` is modeled as a "producer" vertex with
        an edge from SOURCE to the producer node (itself) with capacity `v`.
        - A negative value `v` in `vals` is modeled as a "consumer" vertex with
        an edge from the consumer node (itself) to TARGET with capacity `-v`.
        - An edge with unbounded capacity is added from every producer vertex to every consumer vertex.
        - If the sum of producing capacities 'p' is larger than the sum of consuming capacities 'c',
        the edge capacities from SOURCE to producers are multiplied with 'p'/'c' and if 'c' > 'p'
        the analogous modification is done for edges from consumers to TARGET.
        This modification ensures fairness of the resulting flow.

        Parameters:
        vals (pd.Series): A pandas Series containing float values where each index represents a node.

        Returns:
        nx.DiGraph: A directed graph where the nodes and edges are constructed based on `vals`,
                    ready to run a max-flow algorithm from SOURCE to TARGET.

        Example:
        >>> vals = pd.Series([10, -5, 15, -8])
        >>> network = construct_fair_network(vals)
        >>> tradingVolume, flow_dict = nx.maximum_flow(network, SOURCE, TARGET)
        """
        supply: float = self.getSupply
        demand: float = self.getDemand

        # scale supply/demand s.t. both market sides have equal quantity
        supply_ratio: float = 1
        demand_ratio: float = 1
        if supply > demand:
            supply_ratio = demand / supply
        elif demand != 0:
            demand_ratio = supply / demand

        network: nx.DiGraph = nx.DiGraph()
        network.add_node(SOURCE)
        network.add_node(TARGET)
        producers: List[int] = []
        consumers: List[int] = []
        for i, val in enumerate(vals):
            node = self._get_node(i)
            network.add_node(node)
            if val >= 0:
                producers.append(i)
                network.add_edge(SOURCE, node, capacity=val * supply_ratio)
            else:
                consumers.append(i)
                network.add_edge(node, TARGET, capacity=-val * demand_ratio)
        network.add_edges_from(
            (self._get_node(supply), self._get_node(demand), {"capacity": UNBOUNDED})
            for supply in producers
            for demand in consumers
        )
        return network

    def _get_layer(self, node: str, graph: nx.DiGraph) -> int:
        """
        Assigns a layer to each node:
        - 0 for the SOURCE node
        - 1 for nodes directly connected to SOURCE
        - 2 for nodes connected to layer 1 nodes
        - and so on, with TARGET being the highest layer.

        Parameters:
        node (str): The node to get the layer for.
        graph (nx.DiGraph): The flow graph.

        Returns:
        int: The layer number of the node.
        """
        if node == SOURCE:
            return 0
        elif node == TARGET:
            return 3
        else:
            # Use the shortest path length from source to determine the layer
            try:
                return nx.shortest_path_length(graph, source=SOURCE, target=node)
            except nx.NetworkXNoPath:
                return 2  # If no path, assign it to the second layer
