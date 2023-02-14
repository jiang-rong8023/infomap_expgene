#coding: utf-8


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import infomap
import networkx as nx
import random
from sklearn import preprocessing
from IPython.display import display
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pandas.core.frame import DataFrame
from cdlib import NodeClustering
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster

from sklearn import decomposition as skldec #用于主成分分析降维的包



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

    print("从network图构建Infomap 网络：")
    for e in G.edges():
        infomapX.add_link(*e)

    print("使用infomap寻找社区：")
    infomapX.run();
    print(f"找到 {infomapX.num_top_modules} 个模块， 编码长度: {infomapX.codelength}")
    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()
    nx.set_node_attributes(G, values=communities, name='community')

    df_communities = pd.DataFrame(list(communities.items()))
    df_communities.columns = ['node_id', 'module_id']


    return infomapX.num_top_modules,df_communities

def findCommunitiesWithClustering(g_original):
    """
    使用infomap算法对网络进行分区
    返回模块数和找到的模块
    """
    #infomapX = infomap.Infomap("--two-level --directed -N4 -s 4 --preferred-number-of-modules 5")

    g1 = nx.convert_node_labels_to_integers(g_original, label_attribute="name")
    name_map = nx.get_node_attributes(g1, "name")
    coms_to_node = defaultdict(list)



    im = infomap.Infomap("--two-level --directed -N4 -s 4 --preferred-number-of-modules 5")

    im.add_nodes(g_original.nodes)

    for source, target, data in g1.edges(data=True):
        if "weight" in data:
            im.add_link(source, target, data["weight"])
        else:
            im.add_link(source, target)
    im.run()

    for node_id, module_id in im.modules:
        node_name = name_map[node_id]
        coms_to_node[module_id].append(node_name)

    coms_infomap = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms_infomap, g_original, "Infomap", method_parameters={"flags": "--two-level --directed -N4 -s 4 --preferred-number-of-modules 5"}
    )


def printCommunities(G,num_comm):
    for i in range(num_comm):
        view = nx.subgraph_view(G, filter_node=lambda x: G.nodes[x]["community"]==i )
        df_edge_view = nx.to_pandas_edgelist(view)
        df_edge_view.rename(columns={"source":"fromNodeId","target":"toNodeId"},inplace=True)
        df_edge_view['fromNode'] = df_edge_view['fromNodeId'].apply(lambda x: pao1_corr_graph[pao1_corr_graph['fromNodeId'] == x]['fromNode'].values[0])
        df_edge_view['toNode'] = df_edge_view['toNodeId'].apply(lambda x: pao1_corr_graph[pao1_corr_graph['toNodeId'] == x]['toNode'].values[0])
        print("类（module） %d:"% i)
        order=['fromNode','toNode','weight','fromNodeId','toNodeId']
        if(len(df_edge_view) > 0):
            df_edge_view = df_edge_view[order]
        display(df_edge_view)

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
                     color = cmapDark(communities[n])
                     )

    #fig, ax = plt.subplots(figsize=(20, 18))
    #ax.axis("off")
    #plt.figure(3)
    plt.show()


import argparse

parser = argparse.ArgumentParser(description='基因表达数据处理参数:')
parser.add_argument('--moduleNum', default=2, type=int, help='聚类数')
parser.add_argument('--sampleNum', default=200, type=int, help='选取样本数')
parser.add_argument('--filterWeight', default=0.4, type=float, help='样本关系权重(选取大于这个权重的样本关系)')
parser.add_argument('--dataFile',default="data/ku50(1).txt",type=str,help='输入数据路径')
args = parser.parse_args()

module_num =args.moduleNum
sample_num = args.sampleNum
filter_weight = args.filterWeight
data_file = args.dataFile
#parser.add_argument('--height', default=4, type=int, help='height of Cylinder')


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
df = df[:sample_num]

#plt.figure(1)
#聚类所有样本，观察是否有离群值或异常值
Z = hierarchy.linkage(df, method ='ward',metric='euclidean')
hierarchy.dendrogram(Z,labels = df.index)
#plt.show()


#label = cluster.hierarchy.cut_tree(Z,height=0.8)
#label = label.reshape(label.size,)
#plt.figure(2)
sns.clustermap(df,method ='ward',metric='euclidean')
plt.show()

#删除离群值


#计算样本相关矩阵
correlation_mat = df.corr(method='pearson')
#标准归一化
correlation_mat = (correlation_mat-correlation_mat.min())/(correlation_mat.max()-correlation_mat.min())
#sns.heatmap(correlation_mat, annot = True)
#plt.show()

# 把样本相关矩阵转化为网络
# 有三列 fromNode, toNode, weight
pao1_corr_graph = correlation_mat.stack().reset_index()
pao1_corr_graph.columns = ["fromNode", "toNode", "weight"]



# 去除重复数据
pao1_corr_graph = pao1_corr_graph.drop_duplicates()

#删除样本循环
pao1_corr_graph = pao1_corr_graph[pao1_corr_graph["fromNode"] != pao1_corr_graph["toNode"]]
pao1_corr_graph = pao1_corr_graph[pao1_corr_graph["weight"] >=filter_weight]
print("输入模型的基因数据如下:")
print(pao1_corr_graph.shape)
display(pao1_corr_graph.head(100))


le = preprocessing.LabelEncoder()
le.fit(pao1_corr_graph['fromNode'].values)
pao1_corr_graph['fromNodeId'] = le.transform(pao1_corr_graph['fromNode'].values)
pao1_corr_graph['toNodeId'] = le.transform(pao1_corr_graph['toNode'].values)
print("对样本数据进行标签编码后如下:")
display(pao1_corr_graph.head(100))

G =  nx.Graph()
for index,row in pao1_corr_graph.iterrows():
    #print(row['fromNode'], row['toNode'])
    #im.addLink(int(row['fromNodeId']), int(row['toNodeId']),row['weight'])
    G.add_edge(row['fromNodeId'], row['toNodeId'], weight=row['weight'])
#G=nx.karate_club_graph()

print("样本数据infomap聚类过程:")
num_comm,df_communities = findCommunities(G,"--two-level  --weight-threshold 0.9 --directed -N4 -s 4 --preferred-number-of-modules %d"%module_num)
display(df_communities.head(100))

print("样本聚类结果：")
printCommunities(G,num_comm)

print("可视化样本聚类网络(顶点使用样本的标签编码)：")
drawNetwork(G)
#print(pao1_corr_graph[pao1_corr_graph["fromNodeId"] == 0]['fromNode'].values[0])

#plot_degree_dist(G)



#%%
