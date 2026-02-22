from torch_geometric.utils import from_networkx
import networkx as nx

def create_schema_graph(db):
    # Initialize sets for entities and edges
    entity_types = set()
    edge_types = set()

    # Populate entity types and edge types
    for table_name, table in db.table_dict.items():
        entity_types.add(table_name)
        for fk_col, pkey_table in table.fkey_col_to_pkey_table.items():
            edge_types.add((table_name, f"f2p_{fk_col}", pkey_table))
            edge_types.add((pkey_table, f"rev_f2p_{fk_col}", table_name))
    
    # Create PyG graph
    g = nx.MultiDiGraph()
    for et in sorted(entity_types):
        g.add_node(et, node_type=et)
    for et1, et2, et3 in sorted(edge_types):
        g.add_edge(et1, et3, edge_type='__'.join([et1, et2, et3]))
        
    # Initialize the line graph
    L = nx.DiGraph()

    # Map each edge in G to a node in L
    for edge in g.edges(data=True):
        edge_label = edge[2]['edge_type']  # Get the edge label
        L.add_node(edge_label, node_label=edge_label)

    # Add edges in the line graph based on shared endpoints in the original graph
    for edge1 in g.edges(data=True):
        for edge2 in g.edges(data=True):
            edge1_label = edge1[2]['edge_type']
            edge2_label = edge2[2]['edge_type']
            if edge1 != edge2 and edge1[1] == edge2[0]:
                L.add_edge(edge1_label, edge2_label)
                
    data = from_networkx(g)
    data.node_dict = {node: i for i, node in enumerate(data.node_type)}
    data.edge_dict = {edge: i for i, edge in enumerate(data.edge_type)}
    line_data = from_networkx(L)
    line_data.node_dict = {node: i for i, node in enumerate(line_data.node_label)}
    return data, line_data
    

# if __name__ == '__main__':
#     from relbench.base.database import Database
#     import rootutils
#     rootutils.setup_root('.', indicator='.project-root', pythonpath=True)
    
#     from src.definitions import DATA_DIR
#     import os
    
#     db = Database.load(os.path.join(DATA_DIR, 'rel-f1/dbs/default_db'))
#     G, L = create_schema_graph(db)
#     print(G)
#     print(L)
    
    