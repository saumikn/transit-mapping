import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import graph_tool.all as gt
import json
    

def adj_hex_metros(metro_edges, headways):
    df_metro = pd.DataFrame(metro_edges, columns=["a", "b", "t", "r"])
    df_metro["h"] = df_metro.r.apply(lambda x: headways[x]) / 2
    df_metro["h2"] = 1 / df_metro.h
    df_metro["t"] = df_metro.t.astype(float)

    df_metro_groups = df_metro.groupby(["a", "b"])
    metro_combined_edges = (
        (1 / df_metro_groups.h2.sum() + df_metro_groups.t.mean())
        .reset_index()
        .values.astype(float)
    )

    # To avoid divide-by-zero errors, turn infinity into 1 day (86400s)
    metro_combined_edges[:, 2] = np.minimum(metro_combined_edges[:, 2], 86400)
    metro_combined_edges = metro_combined_edges.astype(int)

    return metro_combined_edges

def eval_q(q, trial_index, **kwargs):
    
    # drivers = kwargs.pop('data')
    # num_drivers = 100
    # driver_sum = sum(drivers.values())
    # drivers = {k: max(v * num_drivers / driver_sum, 0.001) for k, v in drivers.items()}
    # kwargs['data'] = drivers
    
    
    res = evaluate(**kwargs)
    q.put({"idx": trial_index, "res": {"job_num": (res, 0)}})

def evaluate(data=None, max_dist=7200, hs=200, orig=None, dest=None, input_type="drivers", vis=False, pct=True, scale=False, network="new"):  
    
    path = f"data/metrostl/{network}"
    
    with open(f"{path}/headways.json", 'r') as f:
        current_headways = json.load(f)
        
    if isinstance(data, np.ndarray):
        assert(len(data)==len(current_headways))
        data = dict(zip(current_headways, data))
        
    df_route_lens = pd.read_csv(f"{path}/processed/route_lens.csv", index_col="route_id")
    if input_type == "headways":
        headways = data
        drivers = {k: df_route_lens.loc[k].trip_len / v for k, v in headways.items()}
    else:
        drivers = data
        headways = {k: df_route_lens.loc[k].trip_len / v for k, v in drivers.items()}

    # if vis:
    #     print(headways)
    #     print(drivers)
    #     print(sum(drivers.values()))

    with np.load(f"{path}/processed/edges_{hs}.npz") as data:
        hex_walks = data["hex_walks"]
        hex_metros = data["hex_metros"]
        
    if vis:
        hexs = gpd.read_file(f"{path}/processed/hexs_{hs}.gpkg")
    else:
        hexs = pd.read_csv(f"{path}/processed/minihexs_{hs}.csv")

    hex_metros = adj_hex_metros(hex_metros, headways)
    all_edges = np.concatenate([hex_walks, hex_metros])
        
    G = gt.Graph(all_edges, hashed=True, eprops=[("weights", "float")])
    vids_to_v = dict(zip(G.vp["ids"], range(G.num_vertices())))
    hexs = hexs[hexs['index'].isin(vids_to_v)]
    hexs['index'] = hexs['index'].apply(lambda x: vids_to_v[x])
    hexs = hexs.set_index('index').sort_index()
    
    _pred_map = G.vertex_index.copy() # We don't care about this, just to speed up runtime
    _dist_map = G.new_vp("double",  np.inf) # We don't care about this, just to speed up runtime

    # p20 = np.array(hexs.POP21)
    # j20 = np.array(hexs.C000)
    # b20 = np.array(hexs.BOTH20)
    
    if orig == None:
        ORIG = np.array(hexs.POP21)
    elif orig == 'priority':
        ORIG = np.array(hexs.POP21 + hexs.minority + hexs.no_car_hh + hexs.low_income + hexs.ada_indiv)
    else:
        ORIG = np.array(hexs[orig])
        
    if dest == None:
        DEST = np.array(hexs.C000)
    elif dest == 'service':
        DEST = np.array(hexs.CE01 + hexs.CE02_CNS17 + hexs.CE02_CNS18)
    else:
        DEST = np.array(hexs[dest])
        
    OSUM, DSUM = ORIG.sum(), DEST.sum()
        
    connections = 0
        
    res = []
    for i in tqdm(hexs.index, total=len(hexs), disable=(not vis)):        
        dist_map, _, reached = gt.shortest_distance(G,
                                             i,
                                             weights=G.ep.weights,
                                             max_dist=max_dist,
                                             return_reached=True,
                                             pred_map=_pred_map,
                                             dist_map=_dist_map,
                                            )
        
        if scale:
            l, h = 1800, 7200
            times = (-1* np.clip(np.array(dist_map.a[reached]), l, h) + h) / (h-l)
        else:
            times = np.array(dist_map.a[reached]) < max_dist
        
        # pr.append(p20[reached].sum())
        # jr.append(j20[reached].sum())
        # br.append(b20[reached].sum())
        
        connection = (ORIG[i] * (DEST[reached] * times).sum()) / OSUM
        if pct:
            connection = (ORIG[i] * (DEST[reached] * times).sum()) / OSUM / DSUM
            connections += connection
            res.append((DEST[reached] * times).sum() / DSUM)
        else:
            connection = (ORIG[i] * (DEST[reached] * times).sum()) / OSUM
            connections += connection
            res.append((DEST[reached] * times).sum())
        
    
    
    hexs["reached"] = res
    hexs["reached_scaled"] = hexs.reached * ORIG

    if vis:
        print(connections.sum())
        return hexs
    
    return connections.sum()