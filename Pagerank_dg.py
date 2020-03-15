
# import some stuff
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



DG = nx.DiGraph()
DG.add_weighted_edges_from([("y", "a", 1), ("a", "y",1),("y","y",1),("a","m",1)]) 
nx.draw(DG,with_labels=True)
plt.show()

nds=list(DG.nodes())

print(nds)


pr=nx.pagerank(DG,alpha=1)
print(pr)
rank_vector=np.array([[*pr.values()]])
best_node=np.argmax(rank_vector)

print("The most popular website is ",(nds[best_node]))


#Made with ❤ By Dilip Gehlot
