import pandas as pd
import numpy as np
import colorsys as cs
import time as tm

import igraph as ig
import leidenalg as community_leiden

import modularity_maximization as community_newman

import networkx as nx
#import networkx.algorithms.community as nx_comm
import community as community_louvain
import matplotlib
import matplotlib.cm as cm
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
        print("\nTempo Louvain (igraph), {} vertices e {} arestas: {:.3f}ms".format(g.vcount(), g.ecount(), 1000*(t1-t0)))

    elif id == 2: # Leiden Igraph -----Leidenalg        
        part = community_leiden.find_partition(g, community_leiden.ModularityVertexPartition)
        #part = g.community_leiden()
        t1 = tm.time()
        print("\nTempo Leiden (igraph), {} vertices e {} arestas: {:.3f}ms".format(g.vcount(), g.ecount(), 1000*(t1-t0)))
    
    elif id == 3: # Louvain Networkx
        part = community_louvain.best_partition(g)
        t1 = tm.time()
        nmbr_comm = len(set(part.values()))
        print("\nTempo Louvain (networkx), {} vertices e {} arestas: {:.3f}ms".format(nx.number_of_nodes(g), nx.number_of_edges(g), 1000*(t1-t0)))
        print("Modularidade obtida: {:.3f} e total de comunidades: {}\n\n".format(community_louvain.modularity(part, g), nmbr_comm))

    elif id == 4: # Newman Modularity_maximization
        #part = community_newman.partition(g)
        part = g.community_leading_eigenvector()
        t1 = tm.time()
        print("\nTempo Newman (igraph), {} vertices e {} arestas: {:.3f}ms".format(g.vcount(), g.ecount(), 1000*(t1-t0)))#.format(nx.number_of_nodes(g), nx.number_of_edges(g), 1000*(t1-t0)))

    if id != 3:
        print("Modularidade obtida: {:.3f} e total de comunidades: {}\n\n".format(part.modularity, len(part)))

    return part 


def plot_igraph(g, part, outfile):
    '''
    # Visual profile of the graph
    visual_style = {}
    visual_style["bbox"] = (400,400)
    visual_style["margin"] = 27
    #vertex_color=[RGB_tuples[x] for x in part.membership]
    #visual_style["color"] = RGB_tuples
    visual_style["vertex_size"] = 9
    visual_style["vertex_label_size"] = 22
    visual_style["edge_curved"] = False

    # Set the layout
    #my_layout = g.layout('kamada_kawai')###_lgl()
    visual_style["layout"] = my_layout
    '''
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
    #visual_style["vertex_label_dist"] = 0
    visual_style["vertex_shape"] = "circle"
    visual_style["edge_color"] = g.es["color"]
    # visual_style["bbox"] = (4000, 2500)
    visual_style["vertex_size"] = 20
    visual_style["vertex_label"] = {}#False
    visual_style["layout"] = layout
    visual_style["bbox"] = (1024, 768)
    visual_style["margin"] = 40
    visual_style["edge_curved"] = False
    #visual_style["edge_label"] = g.es["weight"]

    #for vertex in g.vs():
    #    vertex["label"] = vertex.index

    if part is not None:
        #print(part)
        colors = []
        #colors = [plt.cm.RdYlBu(x/len(part)) for x in part]
        #print(colors)
        for i in range(0, max(part)+1):
            colors.append('%06X' % randint(0, 0xFFFFFF))
        for vertex in g.vs():
            vertex["color"] = str('#') + colors[part[vertex.index]]
        visual_style["vertex_color"] = g.vs["color"]

    g.vs["label"] = None


    ig.plot(g, outfile, **visual_style, mark_groups=True)
    #ig.plot(g, outfile, vertex_color=[plt.cm.RdYlBu(x/len(part)) for x in part.membership], **visual_style, vertex_frame_color="rgba(0%, 0%, 0%, 0%)")

    return #visual_style


def plot_netx(g, part, outfile):
    # Draw the graph w/ Networkx
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(g, pos, node_size=60, cmap=plt.cm.RdYlBu, node_color=list(part.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    #plt.show()
    # color the nodes according to their partition
    #cmap = cm.get_cmap('viridis', max(part.values()) + 1)
    #nx.draw_networkx_nodes(g, pos, part.keys(), node_size=40,
    #                    cmap=cmap, node_color=list(part.values()))
    #nx.draw_networkx_edges(g, pos, alpha=0.5)
    
    plt.savefig(outfile)
    plt.close()
    #plt.show()

    return


def main():
    #infile= 'karate_data.csv'
    #infile = 'drosofila_network.txt'
    infile = 'IMDB_network.txt'
    #infile = 'Celegans_network.txt'
    outfile1 = "Louvain_Igraph_imdb.png"
    outfile2 = "Leiden_Igraph_imdb.png"
    outfile3 = "Louvain_Netx_imdb.png"
    outfile4 = "Newman_Netx_imdb.png"

    g1 = read_igraph(infile).simplify(combine_edges={ "width": "sum" })
    g2 = read_netx(infile)

    part1 = detect_communities(g1, 1).membership
    part2 = detect_communities(g1, 2).membership
    part3 = detect_communities(g2, 3)
    part4 = detect_communities(g1, 4).membership

    # Plot the graphs
    #plot_igraph(g1, part1, outfile1)
    #plot_igraph(g1, part2, outfile2)
    #plot_netx(g2, part3, outfile3)
    #plot_igraph(g1, part4, outfile4)


if __name__ == "__main__":
    main()