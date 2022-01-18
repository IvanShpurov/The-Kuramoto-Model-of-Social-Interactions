import numpy as np
import networkx as nx
from datetime import datetime
import os
from sknetwork.clustering import Louvain
from sknetwork.clustering import modularity as mod_metric
def to_polar(t):
    z_x=np.cos(t)
    z_y=np.sin(t)
    return [z_x,z_y]
def Compute_compromise(*argv):
    K_1=argv[0]
    K_2=argv[1]
    w_1=argv[2]
    w_2=argv[3]
    w_comp=(K_1*w_2+K_2*w_1)/(K_1+K_2)
    return w_comp
def Unit_circle(steps=200, R=1):
    phi=np.linspace(-np.pi,np.pi, steps)
    x=R*np.cos(phi)
    y=R*np.sin(phi)
    return x,y
def Graph_colormap(freq_table, connections, color_step=100):
    dim_1, dim_2=freq_table.shape[0], freq_table.shape[1]
    color_pattern=np.zeros([dim_1, dim_2, 3])
    a=np.amax(freq_table)
    b=np.amin(freq_table)
    freq_ind=np.linspace(a,b, color_step)
    color_ind={freq_ind[i]:rgb_cycle[i] for i in range(freq_ind.shape[0])} #This line maps frequncies to colors in
    arr=np.asarray(list(color_ind.keys()))                        # the rgb cycle
    for i in range(freq_table.shape[1]):
        freq_step=freq_table[:,1]
        for node in connections:
            f=freq_table[node,i]
            if i>10:
                if f==freq_table[node, i-1]:
                    color_pattern[node,i,:]=color_pattern[node,i-1,:] #Don't change colors if the freqquency is the same
                else: 
                    j = (np.abs(arr - f)).argmin()
                    color=color_ind.get(arr[j])
                    color_pattern[node, i, :]=color
            else:
                j = (np.abs(arr - f)).argmin() #This line selects the closest frequency match
                color=color_ind.get(arr[j])
                color_pattern[node, i, :]=color
    return color_pattern
def get_rgb(N_colors=100):
    phi = np.linspace(0, 2*np.pi, N_colors)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((            # Three sinusoids
        .5*(1.+np.cos(phi          )), # scaled to [0,1]
        .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
        .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
    return rgb_cycle
def color_code(freq_table, Graph, rgb_cycle): #Returns colorcode for a  graph for a frequency(t)
    connections=Graph
    color_pattern=[]
    a=np.amax(freq_table)
    b=np.amin(freq_table)
    freq_ind=np.linspace(a,b, 100)
    color_ind={freq_ind[i]:rgb_cycle[i] for i in range(freq_ind.shape[0])} #This line maps frequncies to colors in
    arr=np.asarray(list(color_ind.keys()))                        # the rgb cycle
    for node in connections:
        f=freq_table[node]
        j = (np.abs(arr - f)).argmin() #This line selects the closest frequency match
        color=color_ind.get(arr[j])
        color_pattern.append(color)
    return color_pattern
def Average_degree(graph):  #Gets average degree of the Graph  
    a=list(graph.degree())
    average_degree=sum([a[i][1] for i in range(len(a))])/graph.number_of_nodes()
    return average_degree
def compute_modularity(conn_mats_list):
    modularity=[]
    label_list=[]
    for i in range(len(conn_mats_list)):
        louvain = Louvain()
        weight_aj=np.tril(conn_mats_list[i])
        labels = louvain.fit_transform(weight_aj)
        label_list.append(labels)
        modularity.append(mod_metric(weight_aj, labels))
    return modularity, label_list
def label_sort(labels, conn_matrix):#, conn_mat):
    #new_labels=np.zeros_like(labels)
    new_labels=np.array([])
    #old=0
    cluster_sizes=[]
    for i in range(min(labels), max(labels+1)):
        arr=np.where(labels==i)[0]
        new_labels=np.hstack((new_labels, arr))
        cluster_sizes.append(arr.shape[0])
        #new_labels[old:old+arr.shape[0]]=arr
        #old=arr.shape[0]
    sorted_mat=np.zeros_like(conn_matrix)
    new_labels=new_labels.astype(int)
    np.take(conn_matrix, new_labels, axis=0, out=sorted_mat)
    np.take(sorted_mat, new_labels, axis=1, out=sorted_mat)
    return new_labels, sorted_mat, cluster_sizes
def save_graph(DIR, conn_matrix, activation_pattern):
    '''
    This Functions saves the graph and its frequency evolution over time.
    Created Dirs are timestamped.
    
    '''
    if type(conn_matrix)==list:
        N=conn_matrix[0].shape[0]
        conn_matrix=np.asarray(conn_matrix)
    else:
        N=conn_matrix.shape[0]
    steps=activation_pattern.shape[1]
    now = datetime.now()
    timestamp = now.strftime(r"%d.%m.%Y_%H:%M")
    info=r"N={}_LEN={}_TIME_".format(N, steps)
    info+=timestamp
    DIR=os.path.join(os.getcwd(),DIR, info)
    os.makedirs(DIR)
    np.save(os.path.join(DIR, "graph"), conn_matrix)
    np.savetxt(os.path.join(DIR,"activations"), activation_pattern)
