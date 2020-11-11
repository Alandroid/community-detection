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
import leidenalg as community_leiden

import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from random import randint


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
        part = g.community_multilevel()
        t1 = tm.time()
        print("\nRunning time Louvain (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))

    elif id == 2: # Leiden Igraph       
        part = community_leiden.find_partition(g, community_leiden.ModularityVertexPartition)
        #part = g.community_leiden()
        t1 = tm.time()
        print("\nRunning time Leiden (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))
    
    elif id == 3: # Louvain Networkx
        part = community_louvain.best_partition(g)
        t1 = tm.time()
        nmbr_comm = len(set(part.values()))
        print("\nRunning time Louvain (networkx), {} nodes and {} links: {:.3f}ms"
            .format(nx.number_of_nodes(g), nx.number_of_edges(g), 1000*(t1-t0)))

        print("Modularity: {:.3f} \nNumber of communities: {}\n\n"
            .format(community_louvain.modularity(part, g), nmbr_comm))

    elif id == 4: # Newman Igraph
        part = g.community_leading_eigenvector()
        t1 = tm.time()
        print("\nRunning time Newman (igraph), {} nodes and {} links: {:.3f}ms"
            .format(g.vcount(), g.ecount(), 1000*(t1-t0)))

    if id != 3:
        print("Modularity: {:.3f} \nNumber of communities: {}\n\n"
            .format(part.modularity, len(part)))

    return part 


def plot_igraph(g, part, outfile):
    """
    Draw the graph using Igraph
    """
    if part is not None:
        gcopy = g.copy()
        edges = []
        edges_colors = []
        for edge in g.es():
            if part[edge.tuple[0]] != part[edge.tuple[1]]:
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
    visual_style["vertex_size"] = 20
    visual_style["layout"] = layout
    visual_style["bbox"] = (1024, 768)
    visual_style["margin"] = 40
    visual_style["edge_curved"] = False

    if part is not None:
        colors = []
        for i in range(0, max(part)+1):
            colors.append('%06X' % randint(0, 0xFFFFFF))
        for vertex in g.vs():
            vertex["color"] = str('#') + colors[part[vertex.index]]
        visual_style["vertex_color"] = g.vs["color"]

    g.vs["label"] = None

    ig.plot(g, outfile, **visual_style, mark_groups=True)
    return


def plot_netx(g, part, outfile):
    """
    Draw the graph using Networkx
    """
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))  # 8x8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(g, pos, node_size=60, cmap=plt.cm.RdYlBu, node_color=list(part.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    plt.savefig(outfile)
    plt.close()

    return


def main():

    infile= r'data/Karate_network.txt'                # Linux paths
    #infile = r'data/Drosofila_network.txt'        
    #infile = r'data/IMDB_network.txt'
    #infile = r'data/Celegans_network.txt'

    outfile1 = "Louvain_Igraph_imdb.png"
    outfile2 = "Leiden_Igraph_imdb.png"
    outfile3 = "Louvain_Netx_imdb.png"
    outfile4 = "Newman_Netx_imdb.png"

    g1 = read_igraph(infile).simplify(combine_edges={ "width": "sum" })
    g2 = read_netx(infile)

    part1 = detect_communities(g1, 1).membership    # Louvain Igraph
    part2 = detect_communities(g1, 2).membership    # Leiden Igraph
    part3 = detect_communities(g2, 3)               # Louvain Networkx
    part4 = detect_communities(g1, 4).membership    # Newman Igraph

    plot_igraph(g1, part1, outfile1)
    plot_igraph(g1, part2, outfile2)
    plot_netx(g2, part3, outfile3)
    plot_igraph(g1, part4, outfile4)


if __name__ == "__main__":
    main()