"""
This is the main file that runs all of the community detection algorithms.
As for the v1, the algorithms available for comparison are:

    - Newman's spectral method (Igraph)
    - Louvain Method (NetworkX and Igraph)
    - Leiden Method (Igraph)
    
"""
#! /usr/bin/python

import pandas as pd
import numpy as np
import colorsys as cs
import time as tm

import igraph as ig
import leidenalg as leiden

import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from random import randint

from draw_multilayer import LayeredNetworkGraph
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection



def read_igraph(infile):
    g = ig.Graph.Read_Ncol(infile, directed=False)
    return g


def read_netx(infile):
    g = nx.read_edgelist(infile, nodetype=int, encoding="utf-8")
    print('\n{}\n'.format(nx.info(g)))
    return g


def detect_communities(g, id):
    t0 = tm.time()
    
    if id == 1: # Louvain Igraph
        partition = g.community_multilevel()
        t1 = tm.time()
        print("\nRunning time Louvain (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))

    elif id == 2: # Leiden Igraph       
        partition = leiden.find_partition(g, leiden.ModularityVertexPartition)
        #partition = g.community_leiden()
        t1 = tm.time()
        print("\nRunning time Leiden (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))
    
    elif id == 3: # Louvain Networkx
        partition = community_louvain.best_partition(g)
        t1 = tm.time()
        nmbr_comm = len(set(partition.values()))
        print("\nRunning time Louvain (networkx), {} nodes and {} links: {:.3f}ms"
            .format(nx.number_of_nodes(g), nx.number_of_edges(g), 1000*(t1-t0)))

        print("Modularity: {:.3f} \nNumber of communities: {}\n\n"
            .format(community_louvain.modularity(partition, g), nmbr_comm))

    elif id == 4: # Newman Igraph
        partition = g.community_leading_eigenvector()
        t1 = tm.time()
        print("\nRunning time Newman (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))

    if id != 3:
        print("Modularity: {:.3f} \nNumber of communities: {}\n\n"
            .format(partition.modularity, len(partition)))

    return partition 


def plot_monoscale_igraph(g, partition, outfile, visual_style=None):
    """
    Draw the graph using Igraph
    """
    if visual_style is None: # Then we'll build it here
        if partition is not None:
            gcopy = g.copy()
            edges = []
            edges_colors = []
            for edge in g.es():
                if partition[edge.tuple[0]] != partition[edge.tuple[1]]:
                    edges.append(edge)
                    edges_colors.append("gray")
                else:
                    edges_colors.append("black")
            gcopy.delete_edges(edges)
            layout = gcopy.layout("kk")
            g.es["color"] = edges_colors

        else:
            layout = g.layout("kk")
            g.es["color"] = "gray"

        visual_style = {}
        visual_style["vertex_shape"] = "circle"
        visual_style["edge_color"] = g.es["color"]
        visual_style["vertex_size"] = 10
        visual_style["layout"] = layout
        visual_style["bbox"] = (1024, 768)
        visual_style["margin"] = 40
        visual_style["edge_curved"] = False

        if partition is not None:
            colors = []
            for i in range(0, max(partition)+1):
                colors.append('%06X' % randint(0, 0xFFFFFF))
            for vertex in g.vs():
                vertex["color"] = str('#') + colors[partition[vertex.index]]
            visual_style["vertex_color"] = g.vs["color"]

    g.vs["label"] = None

    ig.plot(g, outfile, **visual_style, mark_groups=True)
    return visual_style


def plot_monoscale_netx(g, partition, outfile):
    """
    Draw the graph using Networkx
    """
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))  # 8x8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(g, pos, node_size=60, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    plt.savefig(outfile)
    plt.close()


def igraph_to_netx(igraph_g, directed=False):
    """
    Convert graph from igraph to networkX
    """
    A = igraph_g.get_edgelist()
    if directed:
        G = nx.DiGraph(A) # In case your graph is directed
    else:
        G = nx.Graph(A) # In case you graph is undirected

    return G


def coarsen_nodes_igraph(g, partition, iterator, visual_style):
    #pos = nx.spring_layout(g)
    gcopy = g.copy()
    layout = list(gcopy.layout("kk"))

    partition_set = set(partition)
    #layout = sorted(set(map(tuple, layout)), reverse=True)
    print(layout[33][0])

    ### Supondo que os indices na particao e no layout sejam os mesmos -- aparentemente falso ???
    ### Como manter as arestas entre supernós (e ja antecipando o próximo passo de ligação) ???
    ### Como fazer o tracejadinho ligando nós com seus supernós ???

    g_coarsened = ig.Graph()
    g_coarsened.add_vertices(len(list(partition_set)))

    new_layout = []
    colors = []
    multiplier = []
    # Supernode position is the centroid for that community
    for p in range(len(partition_set)):
        cont = 0
        centroid_x = centroid_y = 0
        for i in range(len(partition)):
            if partition[i] == p:
                centroid_x += layout[i][0]
                centroid_y += layout[i][1]
                color = visual_style["vertex_color"][i]
                print(color)
                cont += 1

        print(cont)
        node_pos = [centroid_x/cont, centroid_y/cont]
        print(node_pos)
        new_layout.append(node_pos)
        colors.append(color)
        multiplier.append(cont)

    visual_style["layout"] = new_layout
    visual_style["vertex_color"] = colors
    visual_style["vertex_size"] = list(visual_style["vertex_size"]*np.array(multiplier))

    plot_monoscale_igraph(g_coarsened, partition, 
        "Louvain_multiscale_{}_coarsened.png".format(iterator), visual_style)


def generate_multiscale_louvain(g):
    ofile = "Louvain_multiscale"

    optimiser = leiden.Optimiser()
    partition = leiden.ModularityVertexPartition(g)
    partition_agg = partition.aggregate_partition()

    i = 0
    while optimiser.move_nodes(partition_agg) > 0:
        i += 1
        partition.from_coarse_partition(partition_agg)
        #G = igraph_to_netx(partition)
        print(partition._membership)
        


        partition_agg = partition_agg.aggregate_partition()
        visual_style = plot_monoscale_igraph(g, partition._membership, ofile + "_{}.png".format(i))
        coarsen_nodes_igraph(g, partition._membership, i, visual_style)
        print("teste")











def plot_multiscale_networkx(outfile):
    # define graphs
    n = 5
    g = nx.erdos_renyi_graph(4*n, p=0.1)
    h = nx.erdos_renyi_graph(3*n, p=0.2)
    i = nx.erdos_renyi_graph(2*n, p=0.4)

    node_labels = {nn : str(nn) for nn in range(4*n)}

    # initialise figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    LayeredNetworkGraph([g, h, i], node_labels=node_labels, ax=ax, layout=nx.spring_layout)
    ax.set_axis_off()
    #plt.savefig(outfile)
    plt.show()


def main():

    infile= r'data/Karate_network.txt'                # Linux paths
    #infile = r'data/Celegans_network.txt'
    #infile = r'data/Drosofila_network.txt'        
    #infile = r'data/IMDB_network.txt'
    

    outfile1 = "Louvain_Igraph_imdb.png"
    outfile2 = "Leiden_Igraph_imdb.png"
    outfile3 = "Louvain_Netx_imdb.png"
    outfile4 = "Newman_Netx_imdb.png"

    g1 = read_igraph(infile).simplify(combine_edges={ "width": "sum" })
    g2 = read_netx(infile)

    #partition1 = detect_communities(g1, 1).membership    # Louvain Igraph
    #partition2 = detect_communities(g1, 2).membership    # Leiden Igraph
    #partition3 = detect_communities(g2, 3)               # Louvain Networkx
    #partition4 = detect_communities(g1, 4).membership    # Newman Igraph

    #_ = plot_monoscale_igraph(g1, partition1, outfile1)
    #_ = plot_monoscale_igraph(g1, partition2, outfile2)
    #plot_monoscale_netx(g2, partition3, outfile3)
    #_ = plot_monoscale_igraph(g1, partition4, outfile4)

    #### Generating the multiscale versions of the networks ####
    generate_multiscale_louvain(g1)

    outfile = "multiscale_netx.png"
    #plot_multiscale_networkx(outfile)
    

if __name__ == "__main__":
    main()