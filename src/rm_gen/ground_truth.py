from pathlib import Path

import cv2
from .core import FeatureAcquisitionStep
# from .ground_truth.sionna_radiomap import generate_radiomap_on_grids
from rasterio.features import rasterize
from rasterio.transform import rowcol
import numpy as np
import typing

from .log_utils import _LoggerWithTQDM

import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

def _run_sionna_scene(
    scene: sionna.rt.Scene,
    results_path: str,
    grid_idx: int,
    min_path_gain: float,
    max_path_gain: float,
    bounding_box: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
    resolution: typing.Tuple[float, float],
    tx_positions: typing.List[typing.Tuple[float, float, float]],
    logger: _LoggerWithTQDM,
    num_rays: int = 10**8,
    num_interactions: int = 2,
    batch_size: int = 10,
):
    rm_solver = RadioMapSolver()
    range_x = bounding_box[1][0] - bounding_box[0][0]
    range_y = bounding_box[1][1] - bounding_box[0][1]
    size = [float(range_x), float(range_y)]
    center_x = (bounding_box[0][0] + bounding_box[1][0]) / 2
    center_y = (bounding_box[0][1] + bounding_box[1][1]) / 2
    center = [float(center_x), float(center_y), 1.5]

    tx_id = 0
    windows_start_i = list(range(0, len(tx_positions), batch_size))
    with logger.tqdm_progress_bar(windows_start_i, desc="Different tx positions") as pbar:
        for wind_s in pbar:
            tx_to_consider = tx_positions[wind_s:wind_s+batch_size]
            for j, tx_pos in enumerate(tx_to_consider):
                # Define and add a first transmitter to the scene
                tx = Transmitter(name=f"tx{j}",
                                    position=[
                                        float(tx_pos[0]),
                                        float(tx_pos[1]),
                                        float(tx_pos[2]),
                                    ],
                                    orientation=[0, 0, 0],
                                    power_dbm=23)
                scene.add(tx)

            rm = rm_solver(scene,
                            max_depth=num_interactions,           # Maximum number of ray scene interactions
                            samples_per_tx=num_rays, # If you increase: less noise, but more memory required
                            cell_size=resolution,      # Resolution of the radio map
                            center=center,         # Center of the radio map
                            size=size,             # Total size of the radio map
                            orientation=[0, 0, 0], # Orientation of the radio map, e.g., could be also vertical
                            refraction=False,
                            specular_reflection=True,
                            diffuse_reflection=False,
                            diffraction=True,
                        )

            with np.errstate(divide='ignore', invalid='ignore'):
                gain = 10 * np.log10(rm.path_gain)

            for j, tx_pos in enumerate(tx_to_consider):
                norm_gain = np.copy(gain[j])
                min_g = min_path_gain
                max_g = max_path_gain
                norm_gain[norm_gain < min_g] = min_g
                norm_gain[norm_gain > max_g] = max_g
                norm_gain = (norm_gain - min_g) / (max_g-min_g)

                norm_gain = (norm_gain * 255).astype(np.uint8)

                cv2.imwrite(f"{results_path}/{grid_idx}_{tx_id}.png", norm_gain)
                tx_id += 1

                scene.remove(f"tx{j}")




class SionnaRadiomapOnGrids(FeatureAcquisitionStep):
    """Generates ground truth radiomap on grids using Sionna.
    """
    def __init__(self,
                 dataset_dir: Path,
                 frequency: float,
                 bandwidth: float,
                 number_of_rays: int,
                 number_of_interactions: int,
                 min_gain: float,
                 max_gain: float):
        super().__init__(dataset_dir, "sionna_ground_truth")
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.number_of_rays = number_of_rays
        self.number_of_interactions = number_of_interactions
        self.min_gain = min_gain
        self.max_gain = max_gain

    def parse_ctx(self, raw_ctx):
        for required_key in [
            "candidate_txs",
            "grid_definitions"
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def cache_keys(self, ctx):
        return [self.feature_name + f"_{self.min_gain}_{self.max_gain}_{self.number_of_interactions}_{self.number_of_rays}_{self.frequency}/"]

    def compute(
        self,
        ctx
    ):
        candidate_txs = ctx["candidate_txs"]
        grid_definitions = ctx["grid_definitions"]
        local2grid = grid_definitions["local2grid"]
        grid2local = grid_definitions["grid2local"]
        resolution = grid_definitions["resolution"]
        scene_file_path = ctx["mitsuba_scene_path"]

        xs = candidate_txs.geometry.x.values
        ys = candidate_txs.geometry.y.values
        zs = candidate_txs["height"].values

        gy, gx = rowcol(local2grid, xs, ys)

        masked_grid_indices = np.array(list(grid_definitions["dangerously_mutable__masked_indices"]))

        all_bboxes = np.array(grid_definitions["bboxes"])
        all_indices = np.arange(len(all_bboxes))
        # unmasked_indices = np.setdiff1d(all_indices, masked_grid_indices)
        # bboxes = all_bboxes[unmasked_indices][:1]

        scene = load_scene(scene_file_path)

        save_folder = self.cache_dirs(self.cache_keys(ctx))[0]
        save_folder.mkdir(parents=True, exist_ok=True)

        scene.frequency = self.frequency
        scene.bandwidth = self.bandwidth

        # Configure antenna arrays for all transmitters and receivers
        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    pattern="iso",
                                    polarization="V")
        scene.rx_array = scene.tx_array
        num_of_txs_per_grid = []

        with self._logger.tqdm_progress_bar(
            enumerate(all_bboxes),
            total=len(all_bboxes),
            desc="Generating Sionna Radiomap on Grids",
        ) as pbar:
            for idx, (bbox_min, bbox_max) in pbar:
                if idx in masked_grid_indices:
                    num_of_txs_per_grid.append(0)
                    continue

                xmin, ymin = bbox_min
                xmax, ymax = bbox_max

                mask = (
                    (gx >= xmin) & (gx < xmax) &
                    (gy >= ymin) & (gy < ymax)
                )
                tx_indices_in_bbox = np.where(mask)[0]
                num_of_txs_per_grid.append(len(tx_indices_in_bbox))

                if tx_indices_in_bbox.size == 0:
                    self._logger.debug(
                        f"No candidate TX in bbox {bbox_min} to {bbox_max}, skipping..."
                    )
                    continue

                tx_x = xs[tx_indices_in_bbox]
                tx_y = ys[tx_indices_in_bbox]
                tx_z = zs[tx_indices_in_bbox]

                local_xmin, local_ymin = grid2local * (xmin, ymin)
                local_xmax, local_ymax = grid2local * (xmax, ymax)

                _run_sionna_scene(
                    scene=scene,
                    results_path=str(save_folder.resolve()),
                    grid_idx=idx,
                    min_path_gain=self.min_gain,
                    max_path_gain=self.max_gain,
                    bounding_box=((local_xmin, local_ymin), (local_xmax, local_ymax)),
                    resolution=resolution,
                    tx_positions=list(zip(tx_x, tx_y, tx_z)),
                    logger=self._logger,
                    num_rays=self.number_of_rays,
                    num_interactions=self.number_of_interactions,
                    batch_size=10,
                )

                folder_in_dataset = save_folder.resolve().relative_to(self._cache_dir.resolve())

        self._logger.info(
            f"Saved ground truth to folder {str(save_folder.resolve())}"
        )

        return {
            self.feature_name: {
                "folder": save_folder,
                "feature_files_for_grid": [
                    [
                        folder_in_dataset / f"{idx}_{tx_k}.png"
                        for tx_k in range(num_of_txs_per_grid[idx])
                    ] for idx in range(len(all_bboxes))
                ],
            },
        }
