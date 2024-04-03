import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import graph_tool.all as gt
from itertools import product, combinations
from collections import Counter
import json

def get_outlines():
    # Get Census Block geometries and populations
    pop = gpd.read_file("data/missouri-shp/stl-cleaned-demo.gpkg")

    pop['COUNTYFP20'] = pop.GEOID20.str.slice(2,5)
    pop = pop[pop.COUNTYFP20.isin(["510", "189"])]  # City=510; County=189
    # pop = pop[pop.COUNTYFP20.isin(["510"])]  # City=510; County=189
    stl_outline_unproj = (
        pop.geometry.unary_union
    )  # Need to generate graph before projection
    pop["geometry"] = pop.geometry.to_crs(crs)  # Project graph so we can do distances
    stl_outline = pop.geometry.unary_union

    return pop, stl_outline_unproj, stl_outline

def get_ox_graph(crs, mps, stl_outline_unproj):
    # Get OpenStreetMap Walking Graph, with walking times for each edge
    G0 = ox.graph_from_polygon(stl_outline_unproj, network_type="walk")
    G0 = ox.project_graph(G0, to_crs=crs)
    for _, _, _, data in G0.edges(data=True, keys=True):
        data["time"] = data["length"] / mps

    walk_edges = np.array([(n1, n2, t) for n1, n2, t in G0.edges(data="time")]).astype(
        int
    )
    G_walk = gt.Graph(walk_edges, hashed=True, eprops=[("weights", "float")])
    return G0, G_walk

def get_hexs(G0, pop, bounding_box, size):
    minx, miny, maxx, maxy = bounding_box
    hexagons = []
    dx, dy = 3 / 2 * size, np.sqrt(3) * size
    x, y = minx, miny
    ij = []
    for i, x in enumerate(np.arange(minx, maxx + size, dx)):
        # Offset if i is even
        for j, y in enumerate(np.arange(miny - (i % 2) * dy / 2, maxy + size, dy)):
            vertices = []
            for k in range(6):
                angle = 2 * np.pi / 6 * k
                vertices.append((x + size * np.cos(angle), y + size * np.sin(angle)))
            hexagons.append(Polygon(vertices))
            ij.append([i, j])
    hexs = gpd.GeoDataFrame(
        np.array(ij), columns=["i", "j"], geometry=hexagons, crs=crs
    )

    demos = ['minority', 'low_income', 'no_car_hh', 'lep', 'young',
             'senior', 'ada_indiv', 'POP21', 'C000', 'CE01', 'CE02',
             'CE03', 'CE02_CNS17', 'CE02_CNS18']
    for demo in demos:
        hexs[demo] = 0.0
    hexs["COUNTYFP20"] = ""
    pop_hex = gpd.sjoin(pop, hexs, how="left", predicate="intersects")

    for _, row in tqdm(pop_hex.iterrows(), total=len(pop_hex)):
        _pop = row.geometry
        _hex = hexs.loc[row.index_right].geometry
        proportion = _pop.intersection(_hex).area / _pop.area
        # Allocate population proportionally
        for demo in demos:
            hexs.loc[row.index_right, demo] += row[f'{demo}_left'] * proportion
        # hexs.loc[row.index_right, "POP20"] += row.POP20_left * proportion
        # hexs.loc[row.index_right, "JOB20"] += row.JOB20_left * proportion
        hexs.loc[row.index_right, "COUNTYFP20"] = row.COUNTYFP20_left
    hexs = hexs[hexs.COUNTYFP20 != ""]
    # hexs["BOTH20"] = hexs["POP20"] + hexs["JOB20"]
    # hexs = hexs[hexs.BOTH20 > 0]
    hexs["nearest"] = ox.nearest_nodes(G0, hexs.centroid.x, hexs.centroid.y)
    # hexs = hexs[hexs.apply(lambda x: x.geometry.contains(Point(G0.nodes[x.nearest]['x'], G0.nodes[x.nearest]['y'])), axis=1)]
    return hexs

def get_hex_walks(G, hexs):
    vids_to_v = dict(zip(G.vp["ids"], range(G.num_vertices())))
    hex_walk_edges = set()
    h2 = hexs.reset_index().set_index(["i", "j"])
    h2index = set(h2.index)
    for index, row in tqdm(hexs.iterrows(), total=len(hexs)):
        i, j = row.i, row.j
        n1 = vids_to_v[row.nearest]
        dists = gt.shortest_distance(G, n1, weights=G.ep.weights, max_dist=1800)
        o = i % 2
        for i2, j2 in [
            (i - 1, j - o),
            (i - 1, j + 1 - o),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - o),
            (i + 1, j + 1 - o),
        ]:
            if (i2, j2) not in h2index:
                continue
            q2 = h2.loc[(i2, j2)]
            n2 = vids_to_v[q2.nearest]
            if dists[n2] <= 1800:
                hex_walk_edges.add((index, q2["index"], dists[n2]))
    return np.array(list(hex_walk_edges))

def get_hex_metros(hexs):
    path = f"data/metrostl/{network}"
    df_stop_times = pd.read_csv(f"{path}/stop_times.txt")
    df_stops = pd.read_csv(f"{path}/stops.txt")
    df_trips = pd.read_csv(f"{path}/trips.txt")
    df_routes = pd.read_csv(f"{path}/routes.txt")
    df = pd.merge(df_stop_times, df_stops, how="left", on="stop_id")
    df = pd.merge(df, df_trips, how="left", on="trip_id")
    df = pd.merge(df, df_routes, how="left", on="route_id")
    df = gpd.GeoDataFrame(
        df.drop(columns=["stop_lon", "stop_lat"]),
        geometry=gpd.points_from_xy(df.stop_lon, df.stop_lat),
        crs="EPSG:4326",
    )
    df = df.to_crs(crs)
    df['service_id'] = df.service_id.astype(str)
    df = df[df.service_id == service]
    df.arrival_time = pd.to_timedelta(df.arrival_time).apply(
        lambda x: x.total_seconds()
    )
    df = df[
        [
            "trip_id",
            "arrival_time",
            "stop_id",
            "route_id",
            "route_short_name",
            "direction_id",
            "shape_id",
            "geometry",
        ]
    ]
    df["route_id"] = df.route_short_name + "_" + df.route_id
    stop_to_hex = {}
    for index, row in tqdm(
        df.groupby("stop_id").head(1).iterrows(), total=len(set(df.stop_id))
    ):
        q = hexs[hexs.contains(row.geometry)]
        if not len(q):
            continue
        stop_to_hex[row.stop_id] = q.index[0]
    deltas = {}
    for _, group in tqdm(df.groupby("trip_id")):
        times, stops, n = list(group.arrival_time), list(group.stop_id), len(group)
        rid = group.iloc[0].route_id
        # Todo - Code for combining routes
        for i in range(n - 1):
            if stops[i] not in stop_to_hex:
                continue
            for j in range(i + 1, n):
                if stops[j] not in stop_to_hex:
                    continue
                tup = (rid, stop_to_hex[stops[i]], stop_to_hex[stops[j]])
                if tup in deltas:
                    deltas[tup].append(times[j] - times[i])
                else:
                    deltas[tup] = [times[j] - times[i]]
    deltas = {k: np.mean(v) for k, v in deltas.items()}
    metro_edges = np.array([(si, sj, t, r) for (r, si, sj), t in deltas.items()])
    df2 = df[df.stop_id.isin(stop_to_hex)].copy()
    df2["trip_len"] = df2.groupby("trip_id").arrival_time.transform(
        lambda group: group.iloc[-1] - group.iloc[0]
    )
    
    # print(df2)
    # print(list(current_headways))
    # print(df2.groupby("route_id").trip_len.max())
    df_route_lens = dict(df2.groupby("route_id").trip_len.max()[list(current_headways)] * 2)
    return metro_edges, df_route_lens

# Get set of stops on each route
def get_route_stops(hex_metros, route_lens):
    r_stops = {r:set(hex_metros[hex_metros[:,3] == r][:,0].astype(int)) for r in route_lens}
    # Get set of routes on each stop
    stop_rs = {}
    for k,v in r_stops.items():
        for s in v:
            if s not in stop_rs:
                stop_rs[s] = [k]
            else:
                stop_rs[s].append(k)
    return r_stops, stop_rs

def get_hex_walk_stops(hex_walks, hex_metros):
    G_walk_hex = gt.Graph(hex_walks, hashed=True, eprops=[("weights", "float")])
    _pred_map = G_walk_hex.vertex_index.copy() # We don't care about this, just to speed up runtime
    _dist_map = G_walk_hex.new_vp("double",  np.inf) # We don't care about this, just to speed up runtime'
    vp_key = list(G_walk_hex.vp.ids.a)
    vids_to_v = dict(zip(G_walk_hex.vp["ids"], range(G_walk_hex.num_vertices())))
    stop_idxs = list(set(hex_metros[:,0].astype(int)))
    stop_idxs = [vids_to_v[i] for i in stop_idxs]

    hex_walk_stops = []
    for i in tqdm(stop_idxs, disable=False):
        dist_map, _ = gt.shortest_distance(G_walk_hex, i, stop_idxs, weights=G_walk_hex.ep.weights, max_dist=max_dist_walk_transfer, pred_map=_pred_map, dist_map=_dist_map)
        _i = vp_key[i]
        for j, d in enumerate(dist_map):
            if d <= max_dist_walk_transfer:
                hex_walk_stops.append([_i, vp_key[stop_idxs[j]], d])
    hex_walk_stops = np.array(hex_walk_stops, dtype=int)
    hex_walk_stops = np.array([row for row in hex_walk_stops if stop_rs[row[0]] != stop_rs[row[1]]])
    
    # G_walk_hex - gt graph of the hex_walks network
    # stop_idxs - List of vertices in G_walk_hex which are on stops
    # hex_walk_stops - Walking distance between each hex, max of max_dist_walk_transfer
    
    return G_walk_hex, stop_idxs, hex_walk_stops

def adj_hex_metros(metro_edges, headways):
    df_metro = pd.DataFrame(metro_edges, columns=["a", "b", "t", "r"])
    df_metro = df_metro[df_metro.r.isin(headways)]
    df_metro["h"] = df_metro.r.apply(lambda x: headways[x])
    df_metro["h2"] = 1 / df_metro.h
    df_metro["t"] = df_metro.t.astype(float)

    df_metro_groups = df_metro.groupby(["a", "b"])
    metro_combined_edges = (
        (1 / df_metro_groups.h2.sum() + df_metro_groups.t.mean())
        .reset_index()
        .values.astype(float)
    )

    # To avoid divide-by-zero errors, turn infinity into large number
    metro_combined_edges[:, 2] = np.minimum(metro_combined_edges[:, 2], 2**32-1)
    metro_combined_edges = metro_combined_edges.astype(int)

    return metro_combined_edges

def eval_combo(S, comb, disable=True):
    headways = np.ones(len(route_lens)) * 2**32-1
    if comb:
        headways[list(comb)] = 60
    headways = dict(zip(route_lens, headways))
    if len(comb) > 1:
        relevant_stops = set.union(*[r_stops[list(route_lens)[i]] for i in comb])
    else:
        relevant_stops = set()
    hex_walk_stops2 = hex_walk_stops[[h[0] in relevant_stops and h[1] in relevant_stops for h in hex_walk_stops]]
    hex_metros2 = adj_hex_metros(hex_metros, headways)
    # print(hex_metros2[:,2].mean())
    G = gt.Graph(np.concatenate([hex_walk_stops2, hex_metros2]), hashed=True, eprops=[("weights", "float")])
    _pred_map = G.vertex_index.copy() # We don't care about this, just to speed up runtime
    _dist_map = G.new_vp("double",  np.inf) # We don't care about this, just to speed up runtime'
    vp_key = list(G.vp.ids.a)

    subcomb = [list(combinations(comb, c)) for c in range(len(comb))]
    if subcomb:
        subcomb = [cj for ci in subcomb for cj in ci]
    
    for i in tqdm(list(G.vertices()), disable=disable):
        dist_map, _ = gt.shortest_distance(G, i, weights=G.ep.weights, max_dist=max_dist_total, pred_map=_pred_map, dist_map=_dist_map)
        _i = vp_key[int(i)]
        for j, d in enumerate(dist_map.a):
            if d > max_dist_total:
                continue
            _j = vp_key[j]
            if _i == _j:
                continue
            if (_i,_j) not in S:
                S[_i,_j] = {comb: d}
                continue
            
            skip = False
            for c in subcomb:
                if c in S[_i,_j] and S[_i,_j][c] <= d:
                    skip = True
                    break
            if skip:
                continue
            # if disable: print(_i, _j, d)
            S[_i,_j][comb] = d
            
def get_min_stops():
    # Get the closest hex-stops to each node by walk, max one per route
    min_stops = {}
    _pred_map = G_walk_hex.vertex_index.copy() # We don't care about this, just to speed up runtime
    _dist_map = G_walk_hex.new_vp("double",  np.inf) # We don't care about this, just to speed up runtime
    vp_key = list(G_walk_hex.vp.ids.a)
    vids_to_v = dict(zip(G_walk_hex.vp["ids"], range(G_walk_hex.num_vertices())))
    for i in tqdm(hexs.index, disable=False):
        min_stops[i] = set()
        if i not in vids_to_v:
            continue
        _i = vids_to_v[i]
        dist_map, _ = gt.shortest_distance(G_walk_hex, _i, stop_idxs, weights=G_walk_hex.ep.weights,
                                           max_dist=max_dist_walk_od, pred_map=_pred_map, dist_map=_dist_map)
        js = {vp_key[stop_idxs[j]]:d for j,d in enumerate(dist_map) if d <= max_dist_walk_od}
        for k,v in r_stops.items():
            stop_dists = [(js[r], r) for r in v if r in js]
            if stop_dists:
                min_stop = min(stop_dists)
                min_stops[i].add((min_stop[1], min_stop[0]))
    min_stops = {k:v for k,v in min_stops.items() if v}
    return min_stops

            
def get_S_arr():
    n = len(route_lens)
    S = {}
    combs = []
    # combs += list(combinations(range(n),0))
    combs += list(combinations(range(n),1))
    # combs += list(combinations(range(n),2))
    # combs += list(combinations(range(n),3))
    
    for comb in tqdm(combs):
        eval_combo(S, comb)

    s_count = Counter()
    for k, v in tqdm(S.items()):
        s_count[(len(v))] += 1

    S_arr0 = {k:[] for k in s_count}
    S_arr1 = {k:[] for k in s_count}
    S_arr2 = {k:[] for k in s_count}
    for k,v in tqdm(S.items()):
        S_arr0[len(v)].append('_'.join([str(i) for i in k])) # Node pair
        S_arr1[len(v)].append([])
        S_arr2[len(v)].append([])
        for k2,v2 in v.items():
            k2_adj = [i+1 for i in k2] + [0,0,0]
            k2_adj = k2_adj[:3]
            S_arr1[len(v)][-1].append(k2_adj) # Buses used
            S_arr2[len(v)][-1].append(v2) # Times

    for k in tqdm(s_count):
        S_arr0[k] = np.array(S_arr0[k])
        S_arr1[k] = np.array(S_arr1[k])
        S_arr2[k] = np.array(S_arr2[k])
        
    return S_arr0, S_arr1, S_arr2

def get_N_arr():
    # Compute N_arr - possible paths between each pair of nodes, subject to a constraint of only reachable pairs


    headways_none = np.array([0] + [300 for _ in current_headways], dtype=int)

    S_label = []
    S_value = []
    for k in S_arr0:
        
        S_label.extend(S_arr0[k])
        S_value.extend((headways_none[S_arr1[k]].sum(axis=2) + S_arr2[k]).min(axis=1))
        
        # S_label.extend(v[0])
        # S_value.extend((headways_none[v[1]].sum(axis=2) + v[2]).min(axis=1))
        # print(k, v[0].shape, v[1].shape, v[2].shape)
    S_label = dict(zip(S_label, range(len(S_label))))
    S_value = np.array(S_value)


    N_arr0 = {}
    N_arr1 = {}
    N_arr2 = {}
    for k1,v1 in tqdm(min_stops.items()):
        for k2, v2 in min_stops.items():
            if k1 == k2:
                continue

            _paths = []
            _times = []

            for s1, d1,  in v1:
                for s2, d2 in v2:
                    if f'{s1}_{s2}' not in S_label:
                        continue
                    d0 = S_value[S_label[f'{s1}_{s2}']]
                    if d0+d1+d2 > max_dist_total:
                        continue
                    _paths.append(S_label[f'{s1}_{s2}'])
                    _times.append(d1+d2)

            lp = len(_paths)
            if lp == 0:
                continue
            if lp not in N_arr0:
                N_arr0[lp] = [_paths]
                N_arr1[lp] = [_times]
                N_arr2[lp] = [(k1,k2)]
            else:
                N_arr0[lp].append(_paths)
                N_arr1[lp].append(_times)
                N_arr2[lp].append((k1,k2))

    for k in tqdm(N_arr0):
        N_arr0[k] = np.array(N_arr0[k])
        N_arr1[k] = np.array(N_arr1[k])
        N_arr2[k] = np.array(N_arr2[k])
    return N_arr0, N_arr1, N_arr2


if __name__ == '__main__':
    
    import sys
    network = sys.argv[1]
    service = sys.argv[2]
    hs = int(sys.argv[3])
    
    
    crs = "32615"
    mps = 1.38

    path = f"data/metrostl/{network}"
    with open(f"{path}/headways.json", 'r') as f:
        current_headways = json.load(f)

    max_dist_total = 7200
    max_dist_walk_od = 900
    max_dist_walk_transfer = 600
    
    pop, stl_outline_unproj, stl_outline = get_outlines()
    G0, G_walk = get_ox_graph(crs, mps, stl_outline_unproj) # Get walking graph of entire city
    hexs = get_hexs(G0, pop, stl_outline.bounds, hs) # Move all jobs and pops to hexagons
    hex_walks = get_hex_walks(G_walk, hexs) # Get walking times between each hex
    hex_metros, route_lens = get_hex_metros(hexs) # Get transit times between each hex
    
    import pathlib
    pathlib.Path(f"{path}/processed").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(f"{path}/processed/edges_{hs}.npz", hex_walks=hex_walks, hex_metros=hex_metros)
    hexs.to_file(f"{path}/processed/hexs_{hs}.gpkg", index=True)
    hexs.drop(columns=["i", "j", "COUNTYFP20", "nearest", "geometry"]).to_csv(
        f"{path}/processed/minihexs_{hs}.csv", index_label='index'
    )
    pd.DataFrame(list(route_lens.items()), columns=['route_id', 'trip_len']).to_csv(f'{path}/processed/route_lens.csv')
    
    
    # r_stops, stop_rs = get_route_stops(hex_metros, route_lens)
    # G_walk_hex, stop_idxs, hex_walk_stops = get_hex_walk_stops(hex_walks, hex_metros)
    # min_stops = get_min_stops()
    # S_arr0, S_arr1, S_arr2 = get_S_arr()
    # N_arr0, N_arr1, N_arr2 = get_N_arr()
    # np.savez_compressed(f'tmp/S_arr0_{hs}_{network}.npz', **{str(k):v for k,v in S_arr0.items()})
    # np.savez_compressed(f'tmp/S_arr1_{hs}_{network}.npz', **{str(k):v for k,v in S_arr1.items()})
    # np.savez_compressed(f'tmp/S_arr2_{hs}_{network}.npz', **{str(k):v for k,v in S_arr2.items()})
    # np.savez_compressed(f'tmp/N_arr0_{hs}_{network}.npz', **{str(k):v for k,v in N_arr0.items()})
    # np.savez_compressed(f'tmp/N_arr1_{hs}_{network}.npz', **{str(k):v for k,v in N_arr1.items()})
    # np.savez_compressed(f'tmp/N_arr2_{hs}_{network}.npz', **{str(k):v for k,v in N_arr2.items()})