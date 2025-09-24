
############### Modules ###############

# --- Basics ---
import os
import glob
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm, trange
from pympler import asizeof # part of testing procedures for memory usage
from typing import Optional, Tuple, Dict
import tempfile
import re
import traceback


# --- Math and arrays ---
import numpy as np
import xarray as xr
import pandas as pd
from math import hypot

# --- Plotting ---
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText

# --- Efficiency ---
from joblib import Parallel, delayed

# --- FastScape and TopoToolbox ---
import xsimlab as xs
import fastscape # Only needed by the main script but added here for completeness
from fastscape.models import basic_model # Only needed by the main script but added here for completeness
from fastscape.models import sediment_model # Only needed by the main script but added here for completeness
from fastscape.processes import FlowAccumulator # Only needed by the main script but added here for completeness
import topotoolbox as ttb

# --- Network/Graph Analysis ---
import networkx as netx
from networkx.algorithms.tree.branchings import maximum_branching

# --- GeoSpatial Untilities ---
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree



########################## Function And Class Definitions ################################


def flow_analysis(dem_path, flood=False):
    '''
    Function utilizing Topotoolbox to extract river systems from digital elevation models
    '''
    dem = ttb.read_tif(dem_path)

    dem_FO = ttb.FlowObject(dem)

    dem_acc = dem_FO.flow_accumulation()

    dem_strms = ttb.StreamObject(dem_FO, threshold=25, units="m2").klargestconncomps(-1)

    dem_longest_river = dem_strms.klargestconncomps(1)

    dem_basins = dem_FO.drainagebasins()

    if flood:
        dem_graphflood = ttb.run_graphflood(dem)
        return {"DEM":dem, "Flow_Object":dem_FO, "Accumulation":dem_acc, "Streams":dem_strms, "Basins":dem_basins, "Flood": dem_graphflood}

    return {"DEM":dem, "Flow_Object":dem_FO, "Accumulation":dem_acc, "Streams":dem_strms, "Longest_Stream":dem_longest_river, "Basins":dem_basins}

#### Coordinate Transformation Functions Necessary To Map Grid Indices To Graph Coordinates ####

def graph_x_extent(graphs):
    xs = [x for G in graphs.values() for (x, _) in G.nodes]
    if not xs:
        return None
    return (float(min(xs)), float(max(xs)))

def map_centre_to_world_x(centre, *, grid_nx, x0=500000.0, dx=1.0,
                          mode="extent", x_extent=None):
    """
    Convert dyke centre (grid index or world-x) to world-x used by nodes.

    mode="global": x = x0 + centre * dx
    mode="extent": x = xmin + (centre/(grid_nx-1)) * (xmax - xmin)
                   (uses observed graph x-extent)

    World x just refers to a real world CRS required by some Topotoolbox-Functions
    """
    c = float(centre if centre is not None else 0.5*(grid_nx-1))
    if mode == "global":
        return x0 + c * dx
    if mode == "extent" and x_extent and x_extent[0] < x_extent[1]:
        xmin, xmax = x_extent
        x = xmin + (c / max(1.0, (grid_nx - 1))) * (xmax - xmin)
        return max(xmin, min(x, xmax))  # clamp
    return x0 + c * dx


### A river network simulation (steps from streams to graphs and graph metrics) ###

class River_Network_Sim:
    def __init__(self, streams, shp=(101,201)):
        self.streams = streams
        self.shp = shp

    def get_substreams(self):
        self.substreams = {}
        
        labels = self.streams.conncomps()

        for label in labels:
            component_indices = self.streams.node_indices_where(labels == label)
            mask = np.zeros(self.shp, dtype=bool)
            mask[component_indices[0], component_indices[1]] = True
            self.substreams[label] = self.streams.subgraph(mask)

    def coords2graph(self, visualize=False):
        '''
        Function to transform a ttb stream object to a networkx graph object
        '''
        
        self.graphs = {}

        for stream in self.substreams.keys():

            all_coords = self.substreams[stream].xy() 
            lines = [LineString(coords) for coords in all_coords]

            tree = STRtree(lines)
            line_map = {id(geom): i for i, geom in enumerate(lines)}

            G = netx.DiGraph()
            for line in lines:
                coords = list(line.coords)
                for u, v in zip(coords[:-1], coords[1:]):
                    G.add_edge(u, v)

            tolerance = 1e-6
            for line in lines:
                for endpoint in [Point(line.coords[0]), Point(line.coords[-1])]:
                    candidate_indices = tree.query(endpoint.buffer(tolerance), predicate='intersects')
                    for idx in candidate_indices:
                        target = tree.geometries[idx]
                        if target.equals(line):
                            continue
                        if target.distance(endpoint) < tolerance:
                            coords = list(target.coords)
                            for i in range(len(coords) - 1):
                                seg_start = Point(coords[i])
                                seg_end = Point(coords[i + 1])
                                segment = LineString([seg_start, seg_end])
                                if segment.distance(endpoint) < tolerance:
                                    u = tuple(endpoint.coords[0])
                                    snapped = segment.interpolate(segment.project(endpoint)).coords[0]
                                    if not G.has_edge(seg_start.coords[0], snapped):
                                        G.add_edge(seg_start.coords[0], snapped)
                                    if not G.has_edge(snapped, seg_end.coords[0]):
                                        G.add_edge(snapped, seg_end.coords[0])
                                    if not G.has_edge(u, snapped):
                                        G.add_edge(u, snapped)


            try:
                cycles = list(netx.simple_cycles(G))
                for cycle in cycles:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        if G.has_edge(u, v):
                            G.remove_edge(u, v)
                            break  
            except Exception as e:
                print("Error checking cycles:", e)

            self.graphs[stream] = G

        if visualize:
            fig, ax = plt.subplots(figsize=(12, 6))

            graph_sizes = {k: self.graphs[k].number_of_nodes() for k in self.graphs}
            sizes = list(graph_sizes.values())
            cmap = cm.viridis
            norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))

            for k, G in self.graphs.items():
                color = cmap(norm(graph_sizes[k]))
                pos = {n: n for n in G.nodes}
                netx.draw(G, pos, ax=ax, node_size=5, arrows=True, edge_color=color, node_color=color)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(sizes)
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Number of Nodes')
            plt.title("Subgraphs colored by size (node count)")
            plt.show()
        
        return self.graphs


    def compute_outlets(self, dem=None, centre=None, *,
                        x0=500000.0, dx=1.0,
                        centre_grid_nx=None, centre_mapping="extent"):
        """
        Label each subgraph outlet 'W' or 'E' using a consistent midline.
        If a dyke centre is provided, map it to world-x using either:
        - mode='extent': fit [0..grid_nx-1] → observed [xmin..xmax] of the graphs
        - mode='global': x = x0 + centre*dx
        """
        outlet_directions = {}
        grid_nx = centre_grid_nx or self.shp[1] 
        xext = graph_x_extent(self.graphs)      

        if centre is not None:
            mid_x = map_centre_to_world_x(
                centre,
                grid_nx=grid_nx, x0=x0, dx=dx,
                mode=centre_mapping, x_extent=xext
            )
        elif dem is not None:
            # ridge-based fallback if no explicit centre provided
            ridge_ix = int(np.argmax(np.mean(dem, axis=0)))
            mid_x = x0 + ridge_ix * dx
        else:
            # last fallback: middle of observed extent or of the full grid
            if xext:
                mid_x = 0.5 * (xext[0] + xext[1])
            else:
                nx = self.shp[1]
                mid_x = x0 + 0.5 * (nx - 1) * dx

        for k, g in self.graphs.items():
            sinks = [n for n in g.nodes if g.out_degree(n) == 0]
            if not sinks:
                outlet_directions[k] = None
                continue
            x_out, _ = sinks[0]
            outlet_directions[k] = "W" if x_out < mid_x else "E"

        self.outlet_directions = outlet_directions
        self._mid_x = mid_x  
        return outlet_directions



    def compute_metrics(self, metric_list=None):
        """
        Compute graph metrics for river graphs.
        Assumption:
        - G is a directed acyclic graph (DAG) with flow-oriented edges (upstream -> downstream).
        """
        all_metrics = [
            'diameter','avg_path_len','longest_path','num_nodes','num_edges',
            'num_sources','num_sinks','radius','laplacian_spectrum','adjecency_spectrum',
            'algebraic_connectivity','num_connected_components','spectral_radius',
            'spectral_energy','spectral_gap','degree','degree_centrality',
            'betweenness_centrality','closeness_centrality','pagerank','eigenvector',
            'eccentricity','katz_centrality','laplacian_centrality',
            'strahler','shreve','horton'
        ]
        if metric_list is None:
            metric_list = all_metrics


        metrics = {}
        metrics["g_metrics"] = {
            'diameter': {}, 'avg_path_len': {}, 'longest_path': {}, 'num_nodes': {},
            'num_edges': {}, 'num_sources': {}, 'num_sinks': {}, 'radius': {},
            'laplacian_spectrum': {}, 'adjecency_spectrum': {}, 'algebraic_connectivity': {},
            'num_connected_components': {}, 'spectral_radius': {}, 'spectral_energy': {},
            'spectral_gap': {},'horton': {}
        }
        metrics["n_metrics"] = {
            'degree': {}, 'degree_centrality': {}, 'betweenness_centrality': {},
            'closeness_centrality': {}, 'pagerank': {}, 'eigenvector': {},
            'eccentricity': {}, 'katz_centrality': {}, 'laplacian_centrality': {},
            'strahler': {}, 'shreve': {}
        }

        ### Small Helpers ###
        def _edge_len(u, v, G):
            d = G.get_edge_data(u, v, default={})
            if 'length' in d:
                return float(d['length'])
            ux, uy = G.nodes[u].get('xy', (None, None))
            vx, vy = G.nodes[v].get('xy', (None, None))
            if ux is not None and vx is not None:
                return hypot(vx-ux, vy-uy)
            return 1.0

        def _compute_strahler_shreve(G):
            """Return two dicts: strahler[n], shreve[n]."""
            order = {}
            shreve = {}
            for n in netx.topological_sort(G):
                preds = list(G.predecessors(n))
                if len(preds) == 0:
                    order[n] = 1
                    shreve[n] = 1
                else:
                    po = [order[p] for p in preds]
                    m = max(po)
                    order[n] = m + 1 if po.count(m) >= 2 else m
                    shreve[n] = sum(shreve[p] for p in preds)
            return order, shreve

        def _segment_starts_by_order(G, strahler):
            """
            Identify where streams of order ω start.
            Returns dict: starts[ω] = list of nodes that start a stream of order ω.
            """
            from collections import defaultdict
            starts = defaultdict(list)
            for n in G.nodes():
                w = strahler[n]
                preds = list(G.predecessors(n))
                if w == 1 and len(preds) == 0:
                    starts[w].append(n)
                elif w > 1 and len(preds) > 0:
                    po = [strahler[p] for p in preds]
                    m = max(po)
                    if m == w-1 and po.count(m) >= 2:
                        starts[w].append(n)
            return starts

        def _follow_stream_length(G, start, w, strahler):
            """
            Follow the unique downstream path for a stream of order w starting at node 'start'
            until the order increases (>w) or until an outlet is reached. Sum edge lengths.
            """
            length = 0.0
            curr = start
            while True:
                succs = list(G.successors(curr))
                if len(succs) == 0:
                    break  
                nxt = max(succs, key=lambda s: strahler.get(s, 0))
                if strahler.get(nxt, 0) > w:
                    length += _edge_len(curr, nxt, G)
                    break
                length += _edge_len(curr, nxt, G)
                curr = nxt
            return length, curr  


        for k, G in self.graphs.items():

            # ---- counts & DAG-only longest path ----
            if "num_nodes" in metric_list:
                metrics["g_metrics"]['num_nodes'][k] = G.number_of_nodes()
            if "num_edges" in metric_list:
                metrics["g_metrics"]['num_edges'][k] = G.number_of_edges()
            if "num_sources" in metric_list:
                metrics["g_metrics"]['num_sources'][k] = sum(1 for n in G if G.in_degree(n) == 0)
            if "num_sinks" in metric_list:
                metrics["g_metrics"]['num_sinks'][k] = sum(1 for n in G if G.out_degree(n) == 0)
            if "longest_path" in metric_list and netx.is_directed_acyclic_graph(G):
                metrics["g_metrics"]['longest_path'][k] = len(netx.dag_longest_path(G))

            # ---- node metrics ----
            if "degree" in metric_list:
                metrics["n_metrics"]['degree'][k] = dict(netx.degree(G))
            if "degree_centrality" in metric_list:
                metrics["n_metrics"]['degree_centrality'][k] = netx.degree_centrality(G)
            if "betweenness_centrality" in metric_list:
                metrics["n_metrics"]['betweenness_centrality'][k] = netx.betweenness_centrality(G)
            if "closeness_centrality" in metric_list:
                metrics["n_metrics"]['closeness_centrality'][k] = netx.closeness_centrality(G)
            if "pagerank" in metric_list:
                metrics["n_metrics"]['pagerank'][k] = netx.pagerank(G, alpha=0.85)
            if "katz_centrality" in metric_list:
                metrics["n_metrics"]['katz_centrality'][k] = netx.katz_centrality(G)
            if "laplacian_centrality" in metric_list:
                metrics["n_metrics"]['laplacian_centrality'][k] = netx.laplacian_centrality(G)

            # ---- undirected metrics ----
            Gu = G.to_undirected()
            if "diameter" in metric_list:
                metrics["g_metrics"]['diameter'][k] = netx.diameter(Gu)
            if "avg_path_len" in metric_list:
                metrics["g_metrics"]['avg_path_len'][k] = netx.average_shortest_path_length(Gu)
            if "radius" in metric_list:
                metrics["g_metrics"]['radius'][k] = netx.radius(Gu)
            if "eccentricity" in metric_list:
                metrics["n_metrics"]['eccentricity'][k] = netx.eccentricity(Gu)
            if "eigenvector" in metric_list:
                try:
                    metrics["n_metrics"]['eigenvector'][k] = netx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
                except netx.PowerIterationFailedConvergence:
                    metrics["n_metrics"]['eigenvector'][k] = {}
                except Exception:
                    metrics["n_metrics"]['eigenvector'][k] = {}

            # ---- spectra ----
            try:
                lap_spec = np.array(netx.laplacian_spectrum(Gu))
                A = netx.to_numpy_array(Gu, weight=None, dtype=float)
                adj_spec = np.linalg.eigvals(A)
                if "laplacian_spectrum" in metric_list:
                    metrics["g_metrics"]['laplacian_spectrum'][k] = lap_spec
                if "adjecency_spectrum" in metric_list:
                    metrics["g_metrics"]['adjecency_spectrum'][k] = adj_spec

                lap_s = np.sort(lap_spec)
                if "algebraic_connectivity" in metric_list:
                    metrics["g_metrics"]['algebraic_connectivity'][k] = lap_s[1] if len(lap_s) > 1 else 0.0
                if "num_connected_components" in metric_list:
                    metrics["g_metrics"]['num_connected_components'][k] = int(np.sum(np.isclose(lap_s, 0)))
                adj_abs = np.sort(np.abs(adj_spec))[::-1]
                if "spectral_radius" in metric_list:
                    metrics["g_metrics"]['spectral_radius'][k] = adj_abs[0] if len(adj_abs) else 0.0
                if "spectral_energy" in metric_list:
                    metrics["g_metrics"]['spectral_energy'][k] = float(np.sum(np.abs(adj_spec)))
                if "spectral_gap" in metric_list:
                    metrics["g_metrics"]['spectral_gap'][k] = (adj_abs[0] - adj_abs[1]) if len(adj_abs) > 1 else 0.0
            except Exception as e:
                print(f"Spectral metrics failed for graph {k}: {e}")

            ### Strahler, Shreve, Horton ###
            if any(m in metric_list for m in ("strahler","shreve","horton")):
                if not netx.is_directed_acyclic_graph(G):
                    metrics["n_metrics"]['strahler'][k] = {}
                    metrics["n_metrics"]['shreve'][k] = {}
                    metrics["g_metrics"]['horton'][k] = {}
                else:
                    strahler, shreve = _compute_strahler_shreve(G)

                    if "strahler" in metric_list:
                        metrics["n_metrics"]['strahler'][k] = strahler
                    if "shreve" in metric_list:
                        metrics["n_metrics"]['shreve'][k] = shreve

                    if "horton" in metric_list:
                        starts = _segment_starts_by_order(G, strahler)
                        orders = sorted(set(strahler.values()))
                        N = {}
                        L_mean = {}
                        A_mean = {}

                        for w in orders:
                            start_nodes = starts.get(w, [])
                            N[w] = len(start_nodes)
                            seg_lengths = []
                            seg_areas = []

                            for s in start_nodes:
                                Lw, end_node = _follow_stream_length(G, s, w, strahler)
                                seg_lengths.append(Lw)
                                area = G.nodes[end_node].get('acc', None)
                                if area is None:
                                    area = float(shreve[end_node])
                                seg_areas.append(float(area))

                            L_mean[w] = float(np.mean(seg_lengths)) if seg_lengths else np.nan
                            A_mean[w] = float(np.mean(seg_areas))   if seg_areas else np.nan

                        Rb, Rl, Ra = {}, {}, {}
                        for w in orders:
                            w1 = w + 1
                            if w1 in N and N.get(w1, 0) > 0 and N.get(w, 0) > 0:
                                Rb[w] = N[w] / N[w1]
                            else:
                                Rb[w] = np.nan
                            if w1 in L_mean and (L_mean.get(w, np.nan) not in (0, np.nan)) and not np.isnan(L_mean.get(w, np.nan)):
                                Rl[w] = L_mean[w1] / L_mean[w] if (w1 in L_mean and not np.isnan(L_mean[w1]) and L_mean[w] != 0) else np.nan
                            else:
                                Rl[w] = np.nan
                            if w1 in A_mean and (A_mean.get(w, np.nan) not in (0, np.nan)) and not np.isnan(A_mean.get(w, np.nan)):
                                Ra[w] = A_mean[w1] / A_mean[w] if (w1 in A_mean and not np.isnan(A_mean[w1]) and A_mean[w] != 0) else np.nan
                            else:
                                Ra[w] = np.nan

                        metrics["g_metrics"]['horton'][k] = {
                            'orders': orders,
                            'N': N,             # number of streams per order
                            'L_mean': L_mean,   # mean stream length per order
                            'A_mean': A_mean,   # mean (drainage area or Shreve) per order
                            'Rb': Rb, 'Rl': Rl, 'Ra': Ra
                        }

        return metrics


### Bundler class for multiple simulations including an option to compute river network metrics in parallel

def parallel_metrics(k, run, metrics=None, ny=100, nx=200,
                     centre=None, centre_grid_nx=None, centre_mapping="extent"):
    """
    Function to compute river network metrics in parallel
    """
    dem = run.topography__elevation.isel(time=-1).to_numpy()

    x_min = 500000.0
    y_max = 4600000.0
    pixel_size = 1.0
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    crs = CRS.from_epsg(32633)

    with tempfile.TemporaryDirectory(prefix=f"sim_tmp_{k}_") as tmpdir:
        out_fp = Path(tmpdir) / "output.tif"
        with rio.open(str(out_fp), 'w', driver='GTiff',
                      height=dem.shape[0], width=dem.shape[1],
                      count=1, dtype=dem.dtype, crs=crs, transform=transform) as dst:
            dst.write(dem, 1)

        ttb_dict = flow_analysis(str(out_fp), flood=False)

        RNS = River_Network_Sim(ttb_dict["Streams"], shp=(ny, nx))
        RNS.get_substreams()

        result = {
            "graphs":  RNS.coords2graph(visualize=False),
            "outlets": RNS.compute_outlets(
                dem=dem, centre=centre,
                centre_grid_nx=centre_grid_nx, centre_mapping=centre_mapping
            ),
            "metrics": RNS.compute_metrics(metric_list=metrics),
        }
    return result



def build_results_list_from_nc(path, fn, limit=None):
    """
    Builds list from stored nc files from simulations in order to save RAM
    """
    path = Path(path)
    files = sorted(path.glob(f"field_{fn}_*.nc"))

    if limit is not None:
        files = files[:limit]

    datasets = [xr.open_dataset(p) for p in files]
    return datasets

class SIM_Bundler:
    '''
    Class to bundle all simulation runs in one object
    '''
    def __init__(self, ny, nx, results_list=None, results_path=None, fn=None, limit=None):
        self.ny=ny
        self.nx=nx
        if (results_list == None) & (results_path != None):
            self.results_list = build_results_list_from_nc(path=results_path, fn=fn, limit=limit)
        else:
            self.results_list = results_list


    def fastscape2graphs(self, parallel=False, specific_run='all', metrics=None,
                        centres_by_run=None, centre_grid_nx=None, centre_mapping="extent",
                        n_jobs=1):
        if specific_run == 'all':
            specific_run = list(range(len(self.results_list)))

        self.centre_grid_nx = centre_grid_nx or self.nx
        self.centre_mapping = centre_mapping

        def _centre_for_run(idx):
            if isinstance(centres_by_run, dict):
                return centres_by_run.get(idx, None)
            elif isinstance(centres_by_run, (list, tuple, np.ndarray)):
                return centres_by_run[idx]
            else:
                return centres_by_run

        self.runs = {}

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(parallel_metrics)(
                    k, run, metrics=metrics, ny=self.ny, nx=self.nx,
                    centre=_centre_for_run(k),
                    centre_grid_nx=self.centre_grid_nx,
                    centre_mapping=self.centre_mapping
                )
                for k, run in enumerate(self.results_list)
            )
            for k, res in enumerate(results):
                self.runs[k] = res
        else:
            for run_idx in tqdm(specific_run):
                ds  = self.results_list[run_idx]
                dem = ds.topography__elevation.isel(time=-1).to_numpy()
                RNS = River_Network_Sim(ttb_dict["Streams"], shp=(self.ny, self.nx))
                RNS.get_substreams()
                centre_run = _centre_for_run(run_idx)
                self.runs[run_idx] = {
                    "graphs":  RNS.coords2graph(visualize=False),
                    "outlets": RNS.compute_outlets(
                        dem=dem, centre=centre_run,
                        centre_grid_nx=self.centre_grid_nx,
                        centre_mapping=self.centre_mapping
                    ),
                    "metrics": RNS.compute_metrics(metric_list=metrics),
                }


    def compute_dci(self, centres_by_run, *, per_graph=True, return_details=False, tol=0.0):
        """
        Dyke Crossing Index (DCI).

        For each run, look at every subgraph:
        - If the subgraph's outlet is WEST, count it as a crossing if it has at least
            one SOURCE whose x-position is EAST of the dyke centre.
        - If the subgraph's outlet is EAST, count it as a crossing if it has at least
            one SOURCE whose x-position is WEST of the dyke centre.
        """

        def _centre_for_run(run_idx):
            if isinstance(centres_by_run, dict):
                c = centres_by_run.get(run_idx, None)
            elif isinstance(centres_by_run, (list, tuple, np.ndarray)):
                c = centres_by_run[run_idx]
            else:
                c = centres_by_run
            return c

        def _all_x_extents(graphs):
            xs = []
            for G in graphs.values():
                for n in G.nodes:
                    if isinstance(n, (tuple, list)) and len(n) >= 1:
                        xs.append(float(n[0]))
                    else:
                        xy = G.nodes[n].get('xy', None)
                        if xy is not None:
                            xs.append(float(xy[0]))
            if not xs:
                return 0.0, 1.0
            return min(xs), max(xs)

        def _node_x(n, G):
            if isinstance(n, (tuple, list)) and len(n) >= 1:
                return float(n[0])
            xy = G.nodes[n].get('xy', None)
            if xy is None:
                raise ValueError("Cannot determine node x-position; expected tuple node IDs or 'xy' attribute.")
            return float(xy[0])

        results = {}

        for run_idx, run in self.runs.items():
            graphs = run.get("graphs", {})
            outlets = run.get("outlets", {})

            xext = graph_x_extent(graphs)  
            mid_x = map_centre_to_world_x(
                _centre_for_run(run_idx),
                grid_nx=(getattr(self, "centre_grid_nx", None) or self.nx or 1),
                x0=500000.0, dx=1.0,
                mode=(getattr(self, "centre_mapping", "extent")),
                x_extent=xext
            )

            details = {}
            count = 0
            any_cross = False

            for gkey, G in graphs.items():
                out_side = outlets.get(gkey, None)  
                if out_side not in ("W", "E"):
                    details[gkey] = {"outlet": out_side, "crossing": False, "reason": "no_outlet_info"}
                    continue

                sources = [n for n in G.nodes if G.in_degree(n) == 0]
                if not sources:
                    details[gkey] = {"outlet": out_side, "crossing": False, "reason": "no_sources"}
                    continue

                xs = [_node_x(n, G) for n in sources]
                if out_side == "W":  
                    crossing = any(x > mid_x + tol for x in xs)
                else:                 
                    crossing = any(x < mid_x - tol for x in xs)

                details[gkey] = {"outlet": out_side, "crossing": bool(crossing), "mid_x": mid_x}
                if crossing:
                    any_cross = True
                    if per_graph:
                        count += 1

            if not per_graph:
                count = 1 if any_cross else 0

            results[run_idx] = {"DCI_count": count, "details": details} if return_details else count

        return results


### Executions of a metric analysis with plotting utility ###

############# Helpers Tp Get Metrics Which Compare Distributions (The Plots) ############
 
def _clean_numeric(x):
    if x is None:
        return np.array([], dtype=float)
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]

def _ks_2samp(a, b):
    """Return (D, pvalue) using SciPy if available, else a simple fallback (no p)."""
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(a, b, alternative="two-sided", mode="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        x = np.sort(a)
        y = np.sort(b)
        z = np.concatenate([x, y])
        cdf_x = np.searchsorted(x, z, side='right') / x.size
        cdf_y = np.searchsorted(y, z, side='right') / y.size
        D = np.max(np.abs(cdf_x - cdf_y)) if z.size else np.nan
        return float(D), np.nan

def _wasserstein_1d(a, b):
    """Return W1 distance using SciPy if available, else 1D EMD fallback."""
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        a = np.sort(a); b = np.sort(b)
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return np.nan
        grid = np.union1d(a, b)
        Fa = np.searchsorted(a, grid, side='right') / na
        Fb = np.searchsorted(b, grid, side='right') / nb
        diff = np.abs(Fa - Fb)
        return float(np.trapz(diff, grid))

def _mean_abs_pairwise(x, y=None, max_pairs: int = 500_000, rng=None):
    """
    Mean |xi - yj|. If y is None -> within-sample (exclude diagonal in expectation).
    Subsamples pairs if too many to keep memory/time bounded.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    if y is None:
        n = len(x)
        if n < 2: return 0.0
        tot = n*(n-1)//2
        if tot <= max_pairs:
            i, j = np.triu_indices(n, k=1)
        else:
            i = rng.integers(0, n, size=max_pairs)
            j = rng.integers(0, n, size=max_pairs)
            mask = i < j
            if not np.any(mask): mask = np.ones_like(i, dtype=bool)
            i, j = i[mask], j[mask]
        return float(np.mean(np.abs(x[i] - x[j])))
    else:
        y = np.asarray(y, dtype=float)
        n, m = len(x), len(y)
        if n == 0 or m == 0: return np.nan
        tot = n*m
        if tot <= max_pairs:
            return float(np.mean(np.abs(x[:, None] - y[None, :])))
        ii = np.random.default_rng(rng).integers(0, n, size=max_pairs)
        jj = np.random.default_rng(rng).integers(0, m, size=max_pairs)
        return float(np.mean(np.abs(x[ii] - y[jj])))

def _energy_distance_1d(a, b, max_pairs=500_000, rng=None):
    """
    Energy distance estimator in 1D (always ≥0).
    2*E|X-Y| - E|X-X'| - E|Y-Y'|
    """
    if len(a) == 0 or len(b) == 0:
        return np.nan
    e_xy = _mean_abs_pairwise(a, b, max_pairs=max_pairs, rng=rng)
    e_xx = _mean_abs_pairwise(a, None, max_pairs=max_pairs, rng=rng)
    e_yy = _mean_abs_pairwise(b, None, max_pairs=max_pairs, rng=rng)
    return float(2.0*e_xy - e_xx - e_yy)

def two_sample_distances(
    west, east, *, n_boot: int = 1000, alpha: float = 0.05,
    transform=None, rng=None, max_pairs=500_000
) -> Dict[str, Dict[str, float]]:
    """
    Compute KS, Wasserstein-1, and Energy distances (+ bootstrap CIs).
    Returns: {'ks': {'stat', 'pvalue', 'ci_lo','ci_hi'}, 'wass': {...}, 'energy': {...}}
    """
    rng = np.random.default_rng(rng)
    W = _clean_numeric(west)
    E = _clean_numeric(east)
    if transform is not None:
        W = transform(W); E = transform(E)
        W = W[np.isfinite(W)]; E = E[np.isfinite(E)]

    out = {'n_w': int(W.size), 'n_e': int(E.size)}
    if W.size == 0 or E.size == 0:
        out.update({
            'ks': {'stat': np.nan, 'pvalue': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
            'wass': {'stat': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
            'energy': {'stat': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
        })
        return out

    ks_stat, ks_p = _ks_2samp(W, E)
    w1 = _wasserstein_1d(W, E)
    e = _energy_distance_1d(W, E, max_pairs=max_pairs, rng=rng)

    ks_bs, w_bs, e_bs = [], [], []
    for _ in range(int(n_boot)):
        Wb = rng.choice(W, size=W.size, replace=True)
        Eb = rng.choice(E, size=E.size, replace=True)
        ks_bs.append(_ks_2samp(Wb, Eb)[0])
        w_bs.append(_wasserstein_1d(Wb, Eb))
        e_bs.append(_energy_distance_1d(Wb, Eb, max_pairs=max_pairs, rng=rng))
    qlo, qhi = 100*alpha/2, 100*(1-alpha)
    out['ks'] = {'stat': ks_stat, 'pvalue': ks_p,
                 'ci_lo': float(np.percentile(ks_bs, qlo)),
                 'ci_hi': float(np.percentile(ks_bs, qhi))}
    out['wass'] = {'stat': w1,
                   'ci_lo': float(np.percentile(w_bs, qlo)),
                   'ci_hi': float(np.percentile(w_bs, qhi))}
    out['energy'] = {'stat': e,
                     'ci_lo': float(np.percentile(e_bs, qlo)),
                     'ci_hi': float(np.percentile(e_bs, qhi))}
    return out

def _fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) and np.isfinite(hi) else "[NA, NA]"

def metric_analysis(
    SIM, metric, node_based, visualize=True, log=False, bin_num=100,
    annotation=None, show=True,
    compute_stats=True, n_boot=1000, alpha=0.05, transform=None,
    save_stats_path=None
):
    """
    Analyze graph metrics (W/E). Optionally adds KS/Wasserstein/Energy distances with bootstrap CIs.
    """

    param_west, param_east = [], []

    if node_based:
        for run_key, run in SIM.runs.items():
            west_keys = [k for k, v in run["outlets"].items() if v == "W"]
            east_keys = [k for k, v in run["outlets"].items() if v == "E"]
            gnm = run["metrics"]["n_metrics"].get(metric, {})

            p_west = [val for k in gnm if k in west_keys for val in gnm[k].values() if val is not None]
            p_east = [val for k in gnm if k in east_keys for val in gnm[k].values() if val is not None]
            param_west.append(p_west); param_east.append(p_east)
    else:
        for run_key, run in SIM.runs.items():
            west_keys = [k for k, v in run["outlets"].items() if v == "W"]
            east_keys = [k for k, v in run["outlets"].items() if v == "E"]
            gdict = run["metrics"]["g_metrics"].get(metric, {})
            p_west = [gdict[k] for k in gdict if k in west_keys and gdict[k] is not None]
            p_east = [gdict[k] for k in gdict if k in east_keys and gdict[k] is not None]
            param_west.append(p_west); param_east.append(p_east)

    flat_west = [x for sub in param_west for x in sub]
    flat_east = [x for sub in param_east for x in sub]

    if not flat_west and not flat_east:
        print("[metric_analysis] No data found for metric:", metric)
        return {"W": [], "E": [], "stats": None}

    all_vals = (flat_west + flat_east) if flat_east else flat_west
    vmin = float(np.min(all_vals)); vmax = float(np.max(all_vals))
    if np.isclose(vmin, vmax): vmin -= 0.5; vmax += 0.5
    bins = np.linspace(vmin, vmax, max(3, bin_num))

    stats = None
    if compute_stats:
        stats = two_sample_distances(
            flat_west, flat_east,
            n_boot=n_boot, alpha=alpha,
            transform=(np.log10 if (transform == "log10") else None),
            rng=42
        )

    # ---------- Plot ----------
    if visualize:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(22, 6), sharey=True, sharex=True)

        # West
        if flat_west:
            ax[0].hist(flat_west, bins=bins, alpha=0.8)
        else:
            ax[0].text(0.5, 0.5, "No West data", ha="center", va="center", transform=ax[0].transAxes)
        ax[0].set_title("West");  ax[0].set_xlabel(metric); ax[0].set_ylabel("count")
        if log: ax[0].set_yscale("log")

        # East
        if flat_east:
            ax[1].hist(flat_east, bins=bins, alpha=0.8)
        else:
            ax[1].text(0.5, 0.5, "No East data", ha="center", va="center", transform=ax[1].transAxes)
        ax[1].set_title("East");  ax[1].set_xlabel(metric)
        if log: ax[1].set_yscale("log")

        if stats is not None:
            ks_ci = _fmt_ci(stats['ks']['ci_lo'], stats['ks']['ci_hi'])
            wass_ci = _fmt_ci(stats['wass']['ci_lo'], stats['wass']['ci_hi'])
            en_ci = _fmt_ci(stats['energy']['ci_lo'], stats['energy']['ci_hi'])

            pval = stats['ks']['pvalue']
            ptxt = f"{pval:.3g}" if (pval == pval) else "NA"      

            txt = (
                f"n_W={stats['n_w']} | n_E={stats['n_e']}\n"
                f"KS D={stats['ks']['stat']:.3f} {ks_ci}  p={ptxt}\n"
                f"Wass={stats['wass']['stat']:.3f} {wass_ci}\n"
                f"Energy={stats['energy']['stat']:.3f} {en_ci}"
            )

            at = AnchoredText(txt, loc="upper right",
                            prop=dict(size=10), frameon=True, borderpad=0.5)
            at.patch.set_alpha(0.85)
            ax[1].add_artist(at)

        if annotation:
            fig.suptitle(annotation.get("title", ""), y=0.98, fontsize=14)
            if "save_as" in annotation and annotation["save_as"]:
                from pathlib import Path
                Path(annotation["save_as"]).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(annotation["save_as"], dpi=150, bbox_inches="tight")
        if show: plt.show()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.close(fig)

    if compute_stats and save_stats_path:
        s = stats.copy()
        row = {
            "metric": metric,
            "n_w": s['n_w'], "n_e": s['n_e'],
            "ks": s['ks']['stat'], "ks_ci_lo": s['ks']['ci_lo'], "ks_ci_hi": s['ks']['ci_hi'], "ks_p": s['ks']['pvalue'],
            "wasserstein": s['wass']['stat'], "wass_ci_lo": s['wass']['ci_lo'], "wass_ci_hi": s['wass']['ci_hi'],
            "energy": s['energy']['stat'], "energy_ci_lo": s['energy']['ci_lo'], "energy_ci_hi": s['energy']['ci_hi'],
        }
        df_stats = pd.DataFrame([row])
        save_path = Path(save_stats_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".parquet":
            df_stats.to_parquet(save_path, index=False)
        elif save_path.suffix.lower() in (".csv", ".txt"):
            df_stats.to_csv(save_path, index=False)
        else:
            df_stats.to_parquet(save_path.with_suffix(".parquet"), index=False)

    return {"W": flat_west, "E": flat_east, "stats": stats}



### Class to generate an entire row of random fields which will be used as initial topographies for simulation runs

class RandomFieldExperiment:
    '''
    Class that generates random fields, stores the dyke information and executes simulation runs
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def generate_fields(self, centres, thicknesses, kfs, background_kf, path=r"F:\ESPM", name="EXP_test"):
        '''
        Generate fields and store them as nc files to save RAM
        '''
        self.field_dict = {
            "fields": [],
            "centre": [],
            "thickness": [],
            "Kf": [],
            "dist2mean": []
        }

        counter = 0

        for c in centres:
            for tk in thicknesses:
                for kf in kfs:
                    kf_new = (
                        xr.where(
                            (self.x > c - tk/2) & (self.x < c + tk/2),
                            kf, background_kf
                        )
                    )

                    self.field_dict["fields"].append(kf_new)
                    self.field_dict["thickness"].append(tk)
                    self.field_dict["centre"].append(c)
                    self.field_dict["Kf"].append(kf)
                    self.field_dict["dist2mean"].append(
                        np.abs((len(self.x) / 2) - c)
                    )

                    counter += 1

        with open(Path(path) / f"{name}_field_dict.pkl", "wb") as f:
            pickle.dump(self.field_dict, f)

    def run_experiment(self, model, ds_in, n_noises, n_jobs=1, storage=None):
        '''
        Run multiple simulations in parallel with previously generated random topography inits
        '''
        self.results = {}
        self.noises = [np.random.random(self.x.shape) for i in range(0, n_noises)]

        if storage:
            os.makedirs(storage, exist_ok=True)

            def run_model_parallel(model, ds_in, x, y, noise, kf, i, j, storage):
                with model.drop_processes('init_topography'):
                    ds_out = ds_in.xsimlab.update_vars(
                        input_vars={
                            'topography__elevation': noise,
                            'spl__k_coef_bedrock': kf
                        }
                    ).xsimlab.run()

                path = f"{storage}/field_{i}_noise_{j}.nc"
                ds_out.drop_vars("border").to_netcdf(path)
                return path

            for i in trange(len(self.field_dict["fields"])):
                self.results[f"field_{i}"] = Parallel(n_jobs=n_jobs)(
                    delayed(run_model_parallel)(
                        model, ds_in, self.x, self.y,
                        noise=n, kf=self.field_dict["fields"][i], i=i, j=j, storage=storage) for j, n in enumerate(self.noises))

        else:
            def run_model_parallel(model, ds_in, x, y, noise, kf):
                with model.drop_processes('init_topography'):
                    ds_out = ds_in.xsimlab.update_vars(
                        input_vars={
                            'topography__elevation': noise,
                            'spl__k_coef_bedrock': kf
                        }
                    ).xsimlab.run()
                return ds_out

            for i in trange(len(self.field_dict["fields"])):
                self.results[i] = Parallel(n_jobs=n_jobs)(
                    delayed(run_model_parallel)(
                        model, ds_in, self.x, self.y,
                        noise=n, kf=self.field_dict["fields"][i]) for n in self.noises)
                

### Plot the final topography of a random field experiment (only for one noise option)

def plot_sim_result(setup_dict, path, centre, thickness, kf_n, noise=0, visualize=True):
    '''
    Plots a final simulation output topography for a random field experiment
    '''
    centres = setup_dict["centre"]
    thicknesses = setup_dict["thickness"]
    kfs = setup_dict["Kf"]

    fn = None

    for i, (c, t) in enumerate(zip(centres, thicknesses)):
        if (c == int(centre)) and (t == int(thickness)):
            fn = i + kf_n
            break

    path = Path(path)
    files = list(path.glob(f"field_{fn}_noise_{noise}.nc"))
    print(f"Found file_{fn}_noise_{noise}.nc")
    if not files:
        raise FileNotFoundError(f"No file found for pattern field_{fn}_noise_{noise}.nc in {path}")
    file = files[0]

    dataset = xr.open_dataset(file)

    if visualize:
        plt.figure(figsize = (12, 5))
        dataset.topography__elevation.isel(time=-1).plot(cmap = "viridis") #, vmin=0, vmax = 50)
        plt.title(f"Final Topography For Centre {centre}, Thickness {thickness} And Kf {kfs[kf_n]}")
        plt.tight_layout()
        plt.savefig(Path(path) / f"Topo_{int(c)}_{int(t)}_{kfs[kf_n]:.3e}.png")
        plt.show()

    return dataset
                

### Function to automatically generate plot for the results of metric analysis for all simulation results from a random field experiment

##################### Helper To Get The DCI Within One Simulation Analysis Run #####################

def assemble_dci_df(
    SIM,
    centres_by_run,
    setup_dict=None,
    *,
    use_details=True,
    field_id=None,         
    thickness=None,         
    kf=None,                
    centre_field=None       
):
    """
    Collect DCI outcomes + predictors into a tidy DataFrame.
    - If field_id/thickness/kf are provided, they take precedence.
    - Otherwise we try to infer them from SIM.fn and setup_dict.
    """

    dci = SIM.compute_dci(centres_by_run=centres_by_run,
                          per_graph=True, return_details=True)

    fid = field_id if field_id is not None else getattr(SIM, "fn", None)

    if setup_dict is not None and fid is not None:
        if thickness is None and "thickness" in setup_dict:
            try: thickness = setup_dict["thickness"][fid]
            except Exception: pass
        if kf is None and "Kf" in setup_dict:
            try: kf = setup_dict["Kf"][fid]
            except Exception: pass
        if centre_field is None and "centre" in setup_dict:
            try: centre_field = setup_dict["centre"][fid]
            except Exception: pass

    def _centre_for_run(run_idx):
        if isinstance(centres_by_run, dict):
            return centres_by_run.get(run_idx, np.nan)
        if hasattr(centres_by_run, "__len__") and not np.isscalar(centres_by_run):
            return centres_by_run[run_idx] if run_idx < len(centres_by_run) else np.nan
        return centres_by_run

    rows = []

    for run_idx, run in SIM.runs.items():
        ds = SIM.results_list[run_idx]
        src = getattr(ds, "encoding", {}).get("source", "")
        m = re.search(r"noise_(\d+)", src)
        noise_id = int(m.group(1)) if m else run_idx

        centre_run = _centre_for_run(run_idx)

        det = dci[run_idx]["details"]
        if use_details:
            for gkey, info in det.items():
                rows.append({
                    "crossing": int(bool(info["crossing"])),
                    "run_id": f"{fid}_{noise_id}",
                    "field_id": fid,
                    "noise_id": noise_id,
                    "centre": float(centre_run) if np.isscalar(centre_run) else np.nan,
                    "centre_field": float(centre_field) if np.isscalar(centre_field) else np.nan,
                    "thickness": float(thickness) if np.isscalar(thickness) else np.nan,
                    "Kf": float(kf) if np.isscalar(kf) else np.nan,
                })
        else:
            n_graphs = len(run["graphs"])
            n_cross = sum(int(bool(info["crossing"])) for info in det.values())
            rows.append({
                "crossings": n_cross, "trials": n_graphs,
                "run_id": f"{fid}_{noise_id}", "field_id": fid, "noise_id": noise_id,
                "centre": float(centre_run) if np.isscalar(centre_run) else np.nan,
                "centre_field": float(centre_field) if np.isscalar(centre_field) else np.nan,
                "thickness": float(thickness) if np.isscalar(thickness) else np.nan,
                "Kf": float(kf) if np.isscalar(kf) else np.nan,
            })

    return pd.DataFrame(rows)

def _joblib_worker(
    i, centre, thickness, kf, nx, ny, path, metric, node_based,
    log, save_as, bin_num, limit, show, SIM_Bundler_cls, metric_analysis_fn,
    use_matplotlib_aggressively=True,
    setup_dict=None, compute_dci=False, dci_use_details=True,
    centre_grid_nx=None, centre_mapping="extent"   # <— add these
):

    try:
        if use_matplotlib_aggressively:
            matplotlib.use("Agg")

        print(f"[worker] Processing field {i} -> Centre: {centre}, Thickness: {thickness}, KF: {kf}")

        SIM = SIM_Bundler_cls(ny=ny, nx=nx, results_path=path, fn=i, limit=limit)

        SIM.fastscape2graphs(
            parallel=True,
            metrics=metric,
            centres_by_run=centre,                
            centre_grid_nx=centre_grid_nx,         
            centre_mapping=centre_mapping          
        )

        if metric_analysis_fn is not None and metric is not None:
            _ = metric_analysis_fn(
                SIM=SIM,
                metric=metric,
                node_based=node_based,
                visualize=True,
                log=log,
                bin_num=bin_num,
                annotation={
                    "title": f"Centre: {centre}, Thickness: {thickness}, KF: {kf}",
                    "save_as": Path(save_as) / f"{metric}_{int(centre)}_{int(thickness)}_{kf:.3e}.png"
                },
                compute_stats=True, n_boot=1000, alpha=0.05, transform=None,
                show=show,
                save_stats_path=None
            )

        df_dci = None
        if compute_dci:
            df_dci = assemble_dci_df(
                SIM,
                centres_by_run=centre,          
                setup_dict=setup_dict,
                use_details=dci_use_details,
                field_id=i,                     
                thickness=thickness,             
                kf=kf,                           
                centre_field=(centre if np.isscalar(centre) else None)
            )

        return {"i": i, "status": "ok", "df": df_dci}

    except Exception as e:
        return {"i": i, "status": "error", "error": str(e), "traceback": traceback.format_exc(), "df": None}


def simulation_analysis_joblib_callable(
    nx, ny, path, setup_dict, metric, node_based, log, save_as,
    bin_num=50, limit=None, show=False, n_jobs=None, backend="loky",
    SIM_Bundler_cls=None, metric_analysis_fn=None, verbose=10,
    compute_dci=False, dci_use_details=True, save_df_path=None,
    centre_grid_nx=None, centre_mapping="extent",  
    progress="joblib",
):
    """
    Auto-Generate plots/metrics for each random field.
    """

    centres = setup_dict["centre"]
    thicknesses = setup_dict["thickness"]
    kfs = setup_dict["Kf"]

    save_as = Path(save_as)
    save_as.mkdir(parents=True, exist_ok=True)

    fields = list(enumerate(zip(centres, thicknesses, kfs)))
    total = len(fields)
    if n_jobs is None:
        n_jobs = min(total, os.cpu_count() or 1)

    if SIM_Bundler_cls is None:
        raise ValueError("You must pass SIM_Bundler_cls.")

    def _call_parallel():
        return Parallel(
            n_jobs=n_jobs, backend=backend, verbose=(0 if progress=="tqdm" else verbose)
        )(
            delayed(_joblib_worker)(
                i, c, t, k, nx, ny, path, metric, node_based, log, str(save_as),
                bin_num, limit, show, SIM_Bundler_cls, metric_analysis_fn,
                setup_dict=setup_dict, compute_dci=compute_dci, dci_use_details=dci_use_details,
                centre_grid_nx=centre_grid_nx, centre_mapping=centre_mapping
            )
            for i, (c, t, k) in fields
        )

    if progress == "tqdm":
        try:
            from tqdm.auto import tqdm
            from tqdm_joblib import tqdm_joblib
            print(f"Launching joblib.Parallel with backend={backend}, n_jobs={n_jobs}")
            with tqdm_joblib(tqdm(total=total, desc="fields (returned)")):
                results = _call_parallel()
        except Exception:
            print("tqdm_joblib not available; falling back to joblib verbose output.")
            results = _call_parallel()
    elif progress == "joblib":
        print(f"Launching joblib.Parallel with backend={backend}, n_jobs={n_jobs}")
        results = _call_parallel()
    else:
        results = _call_parallel()

    results = sorted(results, key=lambda x: x.get("i", -1))
    oks = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") == "error"]
    print(f"Completed: {len(oks)} OK, {len(errs)} ERROR")

    if errs:
        for r in errs[:5]:
            print(f"Field {r['i']}: {r['error']}")
            print(r['traceback'][:1000])

    dci_df = None
    if compute_dci:
        dfs = [r["df"] for r in oks if r.get("df") is not None]
        import pandas as pd
        dci_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if save_df_path:
            from pathlib import Path
            save_df_path = Path(save_df_path)
            save_df_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if save_df_path.suffix.lower() == ".parquet":
                    dci_df.to_parquet(save_df_path, index=False)  
                elif save_df_path.suffix.lower() in (".csv", ".txt"):
                    dci_df.to_csv(save_df_path, index=False)
                else:
                    dci_df.to_parquet(save_df_path.with_suffix(".parquet"), index=False)
            except Exception as e:
                fallback = save_df_path.with_suffix(".csv")
                print(f"[warning] Parquet save failed ({e}). Falling back to CSV: {fallback}")
                dci_df.to_csv(fallback, index=False)

    return {"results": results, "dci_df": dci_df}