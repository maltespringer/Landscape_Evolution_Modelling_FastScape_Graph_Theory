
############### Modules ###############

import os
import glob
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm, trange
from pympler import asizeof
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
#import scienceplots
#plt.style.use("science")

from joblib import Parallel, delayed

import xarray as xr
import xsimlab as xs
import fastscape
from fastscape.models import basic_model
from fastscape.models import sediment_model
from orographic_precipitation.fastscape_ext import precip_model
import orographic_precipitation
import topotoolbox as ttb
import pyflwdir as pfd

import pyvista as py
import networkx as netx
from networkx.algorithms.tree.branchings import maximum_branching
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import tempfile

import traceback
import matplotlib


########################## Function Definitions ################################


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


    def compute_outlets(self, dem=None, centre=None):
        '''
        Function to compute the outlet direction (west or east) for each simulation
        '''
        outlet_directions = {}

        if centre is not None:
            mid_x = centre / 200
        elif dem is not None:
            mean_elev_per_x = np.mean(dem, axis=0)   
            max_ind = np.argmax(mean_elev_per_x)     
            mid_x = list(range(500000, 500000 + self.shp[1], 1))[max_ind]            
        else:
            all_x = [x for g in self.graphs.values() for x, y in g.nodes]
            min_x = min(all_x)
            max_x = max(all_x)
            mid_x = (min_x + max_x) / 2

        for k, g in self.graphs.items():
            if not netx.is_directed_acyclic_graph(g):
                print(f"Graph {k}: Not a DAG. May have cycles or multiple outlets.")

            sink_nodes = [n for n in g.nodes if g.out_degree(n) == 0]
            if len(sink_nodes) != 1:
                print(f"Graph {k}: Found {len(sink_nodes)} sink nodes. Taking the first.")

            outlet = sink_nodes[0]
            x, y = outlet

            outlet_directions[k] = "W" if x < mid_x else "E"

        self.outlet_directions = outlet_directions
        return outlet_directions


    def compute_metrics(self, metric_list=None):
        '''
        Function to compute graph metrics for river graphs
        '''

        all_metrics = ['diameter', 'avg_path_len', 'longest_path', 'num_nodes', 
                       'num_edges', 'num_sources', 'num_sinks', 'radius',
                       'laplacian_spectrum', 'adjecency_spectrum', 'algebraic_connectivity',
                       'num_connected_components', 'spectral_radius', 'spectral_energy',
                       'spectral_gap', 'degree', 'degree_centrality', 'betweenness_centrality'
                       'closeness_centrality', 'pagerank', 'eigenvector', 'eccentricity',
                       'katz_centrality', 'laplacian_centrality']
        if metric_list is None:
            metric_list = all_metrics

        metrics = {}

        metrics["g_metrics"] = {
            'diameter' : {},
            'avg_path_len' : {},
            'longest_path' : {},
            'num_nodes' : {},
            'num_edges' : {},
            'num_sources' : {},
            'num_sinks' : {},
            'radius' : {},
            'laplacian_spectrum' : {},
            'adjecency_spectrum' : {},
            'algebraic_connectivity': {},
            'num_connected_components': {},
            'spectral_radius': {},
            'spectral_energy': {},
            'spectral_gap': {},
        }
        metrics["n_metrics"] = {
            'degree' : {},
            'degree_centrality': {},
            'betweenness_centrality' : {},
            'closeness_centrality' : {},
            'pagerank' : {},
            'eigenvector' : {},
            'eccentricity' : {},
            'katz_centrality' : {},
            'laplacian_centrality' : {},
        }

        for k, G in self.graphs.items():
            if "num_nodes" in metric_list:
                metrics["g_metrics"]['num_nodes'][k] = G.number_of_nodes()
            if "num_edges" in metric_list:
                metrics["g_metrics"]['num_edges'][k] = G.number_of_edges()
            if "num_sources" in metric_list:
                metrics["g_metrics"]['num_sources'][k] = len([n for n in G.nodes if G.in_degree(n) == 0])
            if "num_sinks" in metric_list:
                metrics["g_metrics"]['num_sinks'][k] = sum(1 for n in G.nodes if G.out_degree(n) == 0)

            if "longest_path" in metric_list:
                if netx.is_directed_acyclic_graph(G):
                    metrics["g_metrics"]['longest_path'][k] = len(netx.dag_longest_path(G))

            #### Node Metrics ####
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


            #### Undirected Metrics ####
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
                    metrics["n_metrics"]['eigenvector'][k] = netx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
                except netx.PowerIterationFailedConvergence:
                    metrics["n_metrics"]['eigenvector'][k] = {}
                except Exception:
                    metrics["n_metrics"]['eigenvector'][k] = {}


            try:
                Gu = G.to_undirected()
                lap_spec = np.array(netx.laplacian_spectrum(Gu))
                A = netx.to_numpy_array(Gu, weight=None, dtype=float)   
                adj_spec = np.linalg.eigvals(A)
                
                if "laplacian_spectrum" in metric_list:
                    metrics["g_metrics"]['laplacian_spectrum'][k] = lap_spec
                if "adjecency_spectrum" in metric_list:
                    metrics["g_metrics"]['adjecency_spectrum'][k] = adj_spec

                lap_spec_sorted = np.sort(lap_spec)
                if "algebraic_connectivity" in metric_list:
                    if len(lap_spec_sorted) > 1:
                        metrics["g_metrics"]['algebraic_connectivity'][k] = lap_spec_sorted[1]
                    else:
                        metrics["g_metrics"]['algebraic_connectivity'][k] = 0
                
                if "num_connected_components" in metric_list:
                    num_zeros = np.sum(np.isclose(lap_spec_sorted, 0))
                    metrics["g_metrics"]['num_connected_components'][k] = int(num_zeros)

                adj_spec_abs_sorted = np.sort(np.abs(adj_spec))[::-1]  

                if "spectral_radius" in metric_list:
                    metrics["g_metrics"]['spectral_radius'][k] = adj_spec_abs_sorted[0] if len(adj_spec_abs_sorted) > 0 else 0
                if "spectral_energy" in metric_list:
                    metrics["g_metrics"]['spectral_energy'][k] = np.sum(np.abs(adj_spec))

                if "spectral_gap" in metric_list:
                    if len(adj_spec_abs_sorted) > 1:
                        spectral_gap = adj_spec_abs_sorted[0] - adj_spec_abs_sorted[1]
                    else:
                        spectral_gap = 0
                    metrics["g_metrics"]['spectral_gap'][k] = spectral_gap

            except Exception as e:
                print(f"Spectral metrics failed for graph {k}: {e}")
        return metrics



### Bundler class for multiple simulations including an option to compute river network metrics in parallel

def parallel_metrics(k, run, metrics=None, ny=100, nx=200, centre=None):
    '''
    Function to compute river network metrics in parallel
    '''
    dem = run.topography__elevation.isel(time=-1).to_numpy()

    # take any coord system, its just needed by ttb
    x_min = 500000.0
    y_max = 4600000.0
    pixel_size = 1.0
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    crs = CRS.from_epsg(32633)

    # Create a private temp dir for this worker; it is removed automatically
    with tempfile.TemporaryDirectory(prefix=f"sim_tmp_{k}_") as tmpdir:
        out_fp = Path(tmpdir) / "output.tif"   # name local to this dir; no collisions

        # write the raster and close it
        with rio.open(
            str(out_fp),
            'w',
            driver='GTiff',
            height=dem.shape[0],
            width=dem.shape[1],
            count=1,
            dtype=dem.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(dem, 1)

        # call the 3rd-party function with the unique path
        ttb_dict = flow_analysis(str(out_fp), flood=False)

        # build River_Network_Sim from ttb_dict
        RNS = River_Network_Sim(ttb_dict["Streams"], shp=(ny, nx))
        RNS.get_substreams()

        result = {
            "graphs": RNS.coords2graph(visualize=False),
            "outlets": RNS.compute_outlets(dem=dem, centre=centre),
            "metrics": RNS.compute_metrics(metric_list=metrics),
        }

    # tempdir and output.tif removed here
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

    def fastscape2graphs(self, parallel=False, specific_run='all', metrics=None, centre=None):
        '''
        Tranforms all stream objects for all simulation runs into graphs and computed their metrics. 
        '''
        if specific_run == 'all':
            specific_run = list(range(0,len(self.results_list), 1))

        self.runs = {}

        if parallel:
            result = Parallel(n_jobs=1)(delayed(parallel_metrics)(k, run, metrics=metrics, ny=self.ny, nx=self.nx, centre=centre) for k, run in enumerate(self.results_list))
            for run, res in enumerate(result):
                self.runs[run] = res
        else:

            for run in tqdm(specific_run):
                dem = self.results_list[run].topography__elevation.isel(time=-1).to_numpy()

                x_min = 500000.0  
                y_max = 4600000.0  
                pixel_size = 1.0

                transform = from_origin(x_min, y_max, pixel_size, pixel_size)

                crs = CRS.from_epsg(32633)

                with rio.open(
                    'output.tif',
                    'w',
                    driver='GTiff',
                    height=dem.shape[0],
                    width=dem.shape[1],
                    count=1,
                    dtype=dem.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(dem, 1)

                ttb_dict = flow_analysis('output.tif', flood=False)

                RNS = River_Network_Sim(ttb_dict["Streams"])
                RNS.get_substreams()

                self.runs[run] = {
                    "graphs" : RNS.coords2graph(visualize=False),
                    "outlets" : RNS.compute_outlets(dem=dem, centre=centre),
                    "metrics" : RNS.compute_metrics(metric_list=metrics),
                }


### Executions of a metric analysis with plotting utility ###

def metric_analysis(SIM, metric, node_based, visualize=True, log=False, bin_num=100, annotation=None, show=True):
    '''
    Function to analyze the graph metrics computed before (seperated by west and east rivers)
    '''
    param_west = []
    param_east = []

    if node_based:
        for run_key, run in SIM.runs.items():
            west_keys = [k for k in run["outlets"].keys() if run["outlets"][k] == "W"]
            east_keys = [k for k in run["outlets"].keys() if run["outlets"][k] == "E"]

            p_west = [
                val
                for k in run["metrics"]["n_metrics"][metric]
                if k in west_keys
                for val in run["metrics"]["n_metrics"][metric][k].values() if val is not None
            ]
            param_west.append(p_west)

            p_east = [
                val
                for k in run["metrics"]["n_metrics"][metric]
                if k in east_keys
                for val in run["metrics"]["n_metrics"][metric][k].values() if val is not None
            ]
            param_east.append(p_east)

    else:
        for run_key, run in SIM.runs.items():
            west_keys = [k for k in run["outlets"].keys() if run["outlets"][k] == "W"]
            east_keys = [k for k in run["outlets"].keys() if run["outlets"][k] == "E"]

            p_west = [run["metrics"]["g_metrics"][metric][k] 
                    for k in run["metrics"]["g_metrics"][metric] 
                    if k in west_keys and run["metrics"]["g_metrics"][metric][k] is not None]
            param_west.append(p_west)

            p_east = [run["metrics"]["g_metrics"][metric][k] 
                    for k in run["metrics"]["g_metrics"][metric] 
                    if k in east_keys and run["metrics"]["g_metrics"][metric][k] is not None]
            param_east.append(p_east)

    flat_west = [item for sublist in param_west for item in sublist]
    flat_east = [item for sublist in param_east for item in sublist]

    if visualize:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(22, 6), sharey=True, sharex=True)

        maximum = max(max(flat_west), max(flat_east))
        minimum = min(min(flat_west), min(flat_east))

        bins = np.linspace(minimum, maximum, bin_num)

        ax[0].hist(flat_west, bins=bins, density=True, edgecolor="black")
        ax[0].set_title("West")

        ax[1].hist(flat_east, bins=bins, density=True, edgecolor="black")
        ax[1].set_title("East")

        if log:
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")

        if annotation is not None:
            plt.suptitle(annotation["title"])
            plt.savefig(annotation["save_as"])

        if show:
            plt.show()
        plt.close()

    return {"W": flat_west, "E": flat_east}


### Class to generate an entire row of random fields which will be used as initial topographies for simulation runs

class RandomFieldExperiment:
    '''
    Class that generates random fields, stores the dyke information and executes simulation runs
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def generate_fields(self, centres, thicknesses, kfs, background_kf, path=r"F:\ESPM"):
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

        with open(Path(path) / "EXP1_field_dict.pkl", "wb") as f:
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

def _joblib_worker(i, centre, thickness, kf, nx, ny, path, metric, node_based,
                   log, save_as, bin_num, limit, show, SIM_Bundler_cls, metric_analysis_fn,
                   use_matplotlib_aggressively=True):
    try:
        if use_matplotlib_aggressively:
            matplotlib.use("Agg")

        print(f"[worker] Processing field {i} -> Centre: {centre}, Thickness: {thickness}, KF: {kf}")

        SIM = SIM_Bundler_cls(ny=ny, nx=nx, results_path=path, fn=i, limit=limit)
        # If inner is parallel, prefer outer threading or switch inner to parallel=False
        SIM.fastscape2graphs(parallel=True, metrics=metric)

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
            show=show
        )
        return {"i": i, "status": "ok"}
    except Exception as e:
        return {"i": i, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def simulation_analysis_joblib_callable(nx, ny, path, setup_dict, metric, node_based, log, save_as,
                                        bin_num=50, limit=None, show=False, n_jobs=None, backend="loky",
                                        SIM_Bundler_cls=None, metric_analysis_fn=None, verbose=10):
    '''
    Auto-Generate plots for each parameter combination of a random field experiment
    '''
    centres = setup_dict["centre"]
    thicknesses = setup_dict["thickness"]
    kfs = setup_dict["Kf"]

    save_as = Path(save_as)
    save_as.mkdir(parents=True, exist_ok=True)

    fields = list(enumerate(zip(centres, thicknesses, kfs)))
    if n_jobs is None:
        n_jobs = min(len(fields), os.cpu_count() or 1)

    if SIM_Bundler_cls is None or metric_analysis_fn is None:
        raise ValueError("You must pass SIM_Bundler_cls and metric_analysis_fn when running from a notebook.")

    print(f"Launching joblib.Parallel with backend={backend}, n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_joblib_worker)(
            i, c, t, k, nx, ny, path, metric, node_based, log, str(save_as),
            bin_num, limit, show, SIM_Bundler_cls, metric_analysis_fn
        )
        for i, (c, t, k) in fields
    )

    results = sorted(results, key=lambda x: x.get("i", -1))
    oks = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") == "error"]
    print(f"Completed: {len(oks)} OK, {len(errs)} ERROR")
    if errs:
        for r in errs[:5]:
            print(f"Field {r['i']}: {r['error']}")
            print(r['traceback'][:1000])
    return results