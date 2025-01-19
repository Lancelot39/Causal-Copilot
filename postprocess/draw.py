from typing import List, Optional, Tuple
import networkx as nx


def _draw_pag_edges(
    dot,
    directed_edges: List[Tuple] = None,
    circle_edges: List[Tuple] = None,
    undirected_edges: List[Tuple] = None,
    bidirected_edges: List[Tuple] = None,
    **attrs,
):
    """Draw the PAG edges.

    PAG edges may have different endpoints.
    """
    # keep track of edges with circular edges between each other because we want to
    # draw edges correctly when there are circular edges
    found_circle_sibs = set()

    # draw all possible causal edges on a graph
    if circle_edges is not None:
        for sib1, sib2 in circle_edges:
            # memoize if we have seen the bidirected circular edge before
            if f"{sib1}-{sib2}" in found_circle_sibs or f"{sib2}-{sib1}" in found_circle_sibs:
                continue
            found_circle_sibs.add(f"{sib1}-{sib2}")

            # set directionality of the edges
            direction = "forward"

            # check if the circular edge is bidirectional
            if (sib2, sib1) in circle_edges:
                direction = "both"
                arrowtail = "odot"
            elif directed_edges is not None and (sib2, sib1) in directed_edges:
                direction = "both"
                arrowtail = "normal"
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(
                sib1,
                sib2,
                arrowhead="odot",
                arrowtail=arrowtail,
                dir=direction,
                color="green",
                **attrs,
            )

    if undirected_edges is not None:
        for neb1, neb2 in undirected_edges:
            neb1, neb2 = str(neb1), str(neb2)
            dot.edge(neb1, neb2, dir="none", color="brown", **attrs)

    if bidirected_edges is not None:
        for sib1, sib2 in bidirected_edges:
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(sib1, sib2, dir="both", color="red", **attrs)

    return dot, found_circle_sibs


def draw(
    G, full_node_names: List[str] = None,
    direction: Optional[str] = None,
    pos: Optional[dict] = None,
    name: Optional[str] = None,
    shape="square",
    **attrs,
):
    """Visualize the graph.

    Parameters
    ----------
    G : pywhy_nx.MixedEdgeGraph
        The mixed edge graph.
    full_node_names: List[str], optional
        The full node names.
    direction : str, optional
        The direction, by default None. See: https://graphviz.org/docs/attrs/rankdir/.
    pos : dict, optional
        The positions of the nodes keyed by node with (x, y) coordinates as values.
        By default None, which will
        use the default layout from graphviz.
    name : str, optional
        Label for the generated graph.
    shape : str
        The shape of each node. By default 'square'. Can be 'circle', 'plaintext'.
    attrs : dict
        Any additional edge attributes (must be strings). For more
        information, see documentation for GraphViz.

    Returns
    -------
    dot : graphviz.Digraph
        DOT language representation of the graph.
    """
    from graphviz import Digraph

    # make a dict to pass to the Digraph object
    g_attr = {"label": name}

    if name is not None:
        dot = Digraph(graph_attr=g_attr)
    else:
        dot = Digraph()

    if pos is not None:
        dot.engine = 'neato'  
        dot.graph_attr['overlap'] = 'false'
        dot.graph_attr['splines'] = 'true'
        dot.graph_attr['K'] = '0.5'

    # set direction from left to right if that's preferred
    if direction == "LR":
        dot.graph_attr["rankdir"] = direction

    circle_edges = None
    directed_edges = None
    undirected_edges = None
    bidirected_edges = None
    if hasattr(G, "circle_edges"):
        circle_edges = G.circle_edges
    if hasattr(G, "directed_edges"):
        directed_edges = G.directed_edges
    if hasattr(G, "undirected_edges"):
        undirected_edges = G.undirected_edges
    if hasattr(G, "bidirected_edges"):
        bidirected_edges = G.bidirected_edges

    # draw PAG edges and keep track of the circular endpoints found
    dot, found_circle_sibs = _draw_pag_edges(
        dot,
        directed_edges,
        circle_edges=circle_edges,
        undirected_edges=undirected_edges,
        bidirected_edges=bidirected_edges,
    )

    # add the nodes that in the G but not in the PAG
    for node in full_node_names:
        if node not in dot.body:
            if len(full_node_names)<=10:
                dot.node(str(node), shape=shape, height=".5", width=".5")

    # get the directed graph component and add any remaining nodes
    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G
    # add any nodes from full_node_names that aren't in directed_G
    for node in full_node_names:
        if node not in directed_G:
            if len(full_node_names)<=10:
                directed_G.add_node(node)

    for v in full_node_names:
        child = str(v)
        if pos and pos.get(v) is not None:
            dot.node(child, shape=shape, height=".5", width=".5", pos=f"{pos[v][0]*10},{pos[v][1]*10}!")
        else:
            dot.node(child, shape=shape, height=".5", width=".5")
        
        try:
            for parent in directed_G.predecessors(v):
                if parent == v or not directed_G.has_edge(parent, v):
                    continue

                # memoize if we have seen the bidirected circular edge before
                if f"{child}-{parent}" in found_circle_sibs or f"{parent}-{child}" in found_circle_sibs:
                    continue
                parent = str(parent)
                if parent == v:
                    dot.edge(parent, child, style="invis", **attrs)
                else:
                    dot.edge(parent, child, color="blue", **attrs)
        except nx.exception.NetworkXError as e:
            # the node is completely independent in the inferred graph
            pass

    return dot
