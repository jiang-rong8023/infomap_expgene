#coding: utf-8


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as colors
import infomap
import networkx as nx
from IPython.display import display
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import sys
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包



def averageDegree(networkx):
    degrees = [val for (node, val) in networkx.degree()]
    sum = 0
    for d in degrees:
        sum += d
    return sum/len(degrees)
def network_info(network, graph = "False"):
    #print("Degree distribution:", nx.degree_histogram(emailNet))
    print("Average degree:", averageDegree(network))
    print("Clustering coefficient:", nx.average_clustering(network))

    for C in (network.subgraph(c) for c in nx.connected_components(network)):
        print("Average Path Length:", nx.average_shortest_path_length(C))
        #break


    if graph:
        #plot deggre centrality
        fig = plt.figure(figsize=(3*1.61803398875, 3))
        ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
        d = np.array(nx.degree_histogram(network))
        y = d / len(network.nodes)
        x = np.arange(len(y))
        ax.plot(x,y,"go")
        ax.set_xlabel("k")
        ax.set_ylabel("Pk")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title("Degree distribution")
        #ax.legend()
        fig.savefig("%s/degree_distribution.pdf" % imagedir,format='pdf' )
        plt.close(fig)




def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.figure(num='Degree Distribution')
    plt.hist(degrees, bins= 240)
    plt.show()

def findCommunities(G,args):
    """
    使用infomap算法对网络进行分区
    返回模块数和找到的模块
    """
    infomapX = infomap.Infomap(args)

    print("Building Infomap network from a graph:")
    for e in G.edges():
        infomapX.add_link(*e)

    print("Find communities with Infomap:")
    infomapX.run();
    print(f"Found  {infomapX.num_top_modules} top modules， codelength: {infomapX.codelength}")
    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()
    nx.set_node_attributes(G, values=communities, name='community')

    df_communities = pd.DataFrame(list(communities.items()))
    df_communities.columns = ['node_id', 'module_id']


    return infomapX.num_top_modules,df_communities



def printCommunities(G,num_comm):
    for i in range(num_comm):
        view = nx.subgraph_view(G, filter_node=lambda x: G.nodes[x]["community"]==i )
        drawModules(view,i)
        plt.clf()
        df_edge_view = nx.to_pandas_edgelist(view)
        df_edge_view.rename(columns={"source":"fromNodeId","target":"toNodeId"},inplace=True)
        df_edge_view['fromNode'] = df_edge_view['fromNodeId'].apply(lambda x: pao1_corr_graph[pao1_corr_graph['fromNodeId'] == x]['fromNode'].values[0])
        df_edge_view['toNode'] = df_edge_view['toNodeId'].apply(lambda x: pao1_corr_graph[pao1_corr_graph['toNodeId'] == x]['toNode'].values[0])
        print("module %d:"% i)
        order=['fromNode','toNode','weight','fromNodeId','toNodeId']
        if(len(df_edge_view) > 0):
            df_edge_view = df_edge_view[order]
        display(df_edge_view)
        df_edge_view.to_csv("%s/module_%s.txt"%(out_dir,i),sep='\t', index=False)


def drawNetwork(G):


    # 位置 map
    pos = nx.spring_layout(G)
    # 模块 id
    communities = [v for k,v in nx.get_node_attributes(G, 'community').items()]
    numCommunities = max(communities) + 1

    cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed', numCommunities)
    cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)

    # 绘制边
    nx.draw_networkx_edges(G, pos,width=0.1)

    # 绘制节点
    nodeCollection = nx.draw_networkx_nodes(G,
                                            pos = pos,
                                            node_color = communities,
                                            cmap = cmapLight

                                            )
    # 设置节点颜色
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)

    #绘制节点标签
    for n in G.nodes():
        plt.annotate(pao1_corr_graph[pao1_corr_graph["fromNodeId"] == n]['fromNode'].values[0],
                     xy = pos[n],
                     textcoords = 'offset points',
                     horizontalalignment = 'center',
                     verticalalignment = 'center',
                     xytext = [0, 0],
                     #color = cmapDark(communities[n])
                     )

    #fig, ax = plt.subplots(figsize=(20, 18))
    #ax.axis("off")
    #plt.figure(num="network")

    plt.tight_layout()
    plt.savefig("./%s/network_all.pdf"%imagedir,format='pdf')
    plt.show()
    plt.clf()

def drawModules(G,moduleNo):


    # 位置 map
    pos = nx.spring_layout(G)
    # 模块 id
    communities = [v for k,v in nx.get_node_attributes(G, 'community').items()]
    numCommunities = max(communities) + 1

    cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed', numCommunities)
    cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)

    # 绘制边
    nx.draw_networkx_edges(G, pos,width=0.1)

    # 绘制节点
    nodeCollection = nx.draw_networkx_nodes(G,
                                            pos = pos,
                                            node_color = communities,
                                            cmap = cmapLight

                                            )
    # 设置节点颜色
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)

    #绘制节点标签
    for n in G.nodes():
        plt.annotate(pao1_corr_graph[pao1_corr_graph["fromNodeId"] == n]['fromNode'].values[0],
                     xy = pos[n],
                     textcoords = 'offset points',
                     horizontalalignment = 'center',
                     verticalalignment = 'center',
                     xytext = [0, 0],
                    # color = cmapDark(communities[n])
                     )

    #fig, ax = plt.subplots(figsize=(20, 18))
    #ax.axis("off")
    #plt.figure(num="network")

    plt.tight_layout()
    plt.savefig("./%s/network_module_%s.pdf"%(imagedir,moduleNo),format='pdf')
    plt.show()
    plt.clf()

import argparse

parser = argparse.ArgumentParser(description='基因表达数据处理参数:')
parser.add_argument('--moduleNum', default=2, type=int, help='聚类数')
parser.add_argument('--trailNum', default=4, type=int, help='迭代次数')
parser.add_argument('--geneNum', default=200, type=int, help='选取基因数')
parser.add_argument('--filterWeight', default=0.4, type=float, help='基因关系权重(选取大于这个权重的基因关系)')
parser.add_argument('--dataFile',default="data/ku50(1).txt",type=str,help='输入数据路径')
parser.add_argument('--outDir',default="gene_exp_infomap_nogui",type=str,help='输出文件目录')
args = parser.parse_args()

module_num =args.moduleNum
gene_num = args.geneNum
filter_weight = args.filterWeight
data_file = args.dataFile
out_dir = args.outDir
trail_num = args.trailNum


#parser.add_argument('--height', default=4, type=int, help='height of Cylinder')
import os

imagedir = out_dir  #sys.argv[0].split('.')[0]
if not os.path.exists(imagedir):
    os.makedirs(imagedir)

#主程序
plt.rcParams.update({
    'figure.figsize': (20, 20),
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False})

df = pd.read_csv(data_file, sep='\t', index_col=0)
#去除重复数据
df =df.drop_duplicates()
#删除缺少行
df = df.dropna()
#取大于0的数据
df = df[(df.T != 0).any()]
#df = df[(df.T > 1).any()]
#display(df.count())
#筛选基因
df = df[:gene_num].T

#plt.figure(1)
#聚类所有样本，观察是否有离群值或异常值,样本聚类树
Z = hierarchy.linkage(df, method ='ward',metric='euclidean')
hierarchy.dendrogram(Z,labels = df.index,color_threshold=3000)
#hierarchy.cut_tree(Z,height=0.8)
plt.savefig("./%s/samples_tree.pdf"%imagedir,format='pdf')
plt.show()
plt.clf()


#label = cluster.hierarchy.cut_tree(Z,height=0.8)
#label = label.reshape(label.size,)
#plt.figure(2)
sns.clustermap(df,method ='ward',metric='euclidean')
plt.savefig("./%s/cluster_map.pdf"%imagedir,format='pdf')
plt.show()
plt.clf()
#plt.close()

#删除离群值


#计算基因相关矩阵
correlation_mat = df.corr(method='pearson')
#标准归一化
correlation_mat = (correlation_mat-correlation_mat.min())/(correlation_mat.max()-correlation_mat.min())
#sns.heatmap(correlation_mat, annot = True)
#plt.show()

# 把基因相关矩阵转化为网络
# 有三列 fromNode, toNode, weight
pao1_corr_graph = correlation_mat.stack().reset_index()
pao1_corr_graph.columns = ["fromNode", "toNode", "weight"]



# 去除重复数据
pao1_corr_graph = pao1_corr_graph.drop_duplicates()

#删除基因循环
pao1_corr_graph = pao1_corr_graph[pao1_corr_graph["fromNode"] != pao1_corr_graph["toNode"]]
def filterw(x):
    if x >= filter_weight:
        return x
    else:
        return 0
    
pao1_corr_graph["weight"] = pao1_corr_graph["weight"].apply(lambda x: filterw(x))
print("Input gene data:")
print(pao1_corr_graph.shape)
display(pao1_corr_graph.head(100))


le = preprocessing.LabelEncoder()
le.fit(pao1_corr_graph['fromNode'].values)
pao1_corr_graph['fromNodeId'] = le.transform(pao1_corr_graph['fromNode'].values)
pao1_corr_graph['toNodeId'] = le.transform(pao1_corr_graph['toNode'].values)
print("Gene data with label encode:")
display(pao1_corr_graph.head(100))

G =  nx.Graph()
for index,row in pao1_corr_graph.iterrows():
    #print(row['fromNode'], row['toNode'])
    #im.addLink(int(row['fromNodeId']), int(row['toNodeId']),row['weight'])
    if(row['weight'] > 0.0):
        G.add_edge(row['fromNodeId'], row['toNodeId'], weight=row['weight'])
#G=nx.karate_club_graph()

print("Gene network info:")
network_info(G)

print("Gene data infomap process:")
num_comm,df_communities = findCommunities(G,"--two-level  --weight-threshold %f --directed -N%d -s 4  --preferred-number-of-modules %d"%(filter_weight,trail_num,module_num))
display(df_communities.head(100))

print("Result of infomap：")
printCommunities(G,num_comm)

print("Display gene network：")
drawNetwork(G)
#print(pao1_corr_graph[pao1_corr_graph["fromNodeId"] == 0]['fromNode'].values[0])





#%%
