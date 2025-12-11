import typing
from pathlib import Path

import cv2
import numpy as np
from rasterio.features import rasterize

from .core import FeatureAcquisitionStep


class FeatureBuildingHeightRasterOnGrids(FeatureAcquisitionStep):
    def __init__(
        self,
        dataset_dir: Path,
        min_h: float,
        max_h: float,
        feature_name: str = "building_height",
        mask_when_perc_of_building_less_than: typing.Optional[float] = None,
    ):
        super().__init__(dataset_dir, feature_name)
        self.min_h = min_h
        self.max_h = max_h
        self.mask_when_perc_of_building_less_than = mask_when_perc_of_building_less_than

    def cache_keys(self, ctx):
        return [self.feature_name + f"_{self.min_h}_{self.max_h}/"]

    def prepare(self, ctx):
        self.clean_cache(ctx)

    def parse_ctx(self, raw_ctx) -> typing.Any:
        for required_key in [
            "local_building_footprints",
            "building_heights",
            "grid_definitions"
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def compute(self, ctx):
        building_fps = ctx["local_building_footprints"]
        building_heights = ctx["building_heights"]
        grid_definitions = ctx["grid_definitions"]
        local2grid = grid_definitions["local2grid"]
        width, height = grid_definitions["local_coord_size"]


        # convert all coordinates to grid coordinates
        # idealizing the api to use rasterio, how would it work?
        normalized_heights = (building_heights - self.min_h) / (self.max_h - self.min_h)

        normalized_heights = np.clip(normalized_heights, 0.0, 1.0)

        normalized_heights_u8 = (normalized_heights * 255).astype(np.uint8)

        raster = np.zeros((height, width), dtype=np.uint8)
        rasterize(
            [(geom, height) for geom, height in zip(building_fps.geometry, normalized_heights_u8)],
            out=raster,
            transform=local2grid
        )

        grid_bboxes = grid_definitions["bboxes"]
        save_folder = self.cache_dirs(self.cache_keys(ctx))[0]
        save_folder.mkdir(parents=True, exist_ok=True)
        grid_idxs_to_mask = []

        folder_in_dataset = save_folder.resolve().relative_to(self._cache_dir.resolve())

        def get_feature_files_for_grid(idx: int) -> str:
            return folder_in_dataset /f"{idx}.png"

        for idx, (bbox_min, bbox_max) in enumerate(grid_bboxes):
            if idx in grid_definitions["dangerously_mutable__masked_indices"]:
                continue

            grid_raster = raster[
                bbox_min[1]:bbox_max[1],
                bbox_min[0]:bbox_max[0],
            ]

            if self.mask_when_perc_of_building_less_than is not None:
                total_building_cells = np.sum(grid_raster > 0)
                if total_building_cells < self.mask_when_perc_of_building_less_than * grid_raster.size:
                    grid_idxs_to_mask.append(idx)
                    grid_definitions["dangerously_mutable__masked_indices"].add(idx)
                    continue
                    
            cv2.imwrite(
                self._cache_dir / get_feature_files_for_grid(idx),
                grid_raster,
            )
        self._logger.debug(
            f"Masked {len(grid_idxs_to_mask)} grids out of {len(grid_bboxes)} total grids"
        )
        self._logger.info(
            f"Saved building height feature grids to folder {str(save_folder.resolve())}"
        )

        return {
            self.feature_name: {
                "masked_grid_indices": grid_idxs_to_mask,
                "folder": save_folder,
                "feature_files_for_grid": [get_feature_files_for_grid(idx) for idx in range(len(grid_bboxes))],
            },
        }


class FeatureTxHeightJsonOnGrids(FeatureAcquisitionStep):
    def __init__(
        self,
        dataset_dir: Path,
        feature_name: str = "tx_height",
        mask_when_num_of_tx_less_than: float = 1,
    ):
        super().__init__(dataset_dir, feature_name)
        self.mask_when_num_of_tx_less_than = mask_when_num_of_tx_less_than

    def cache_keys(self, ctx):
        return [self.feature_name + f"_{self.min_h}_{self.max_h}/"]

    def parse_ctx(self, raw_ctx) -> typing.Any:
        for required_key in [
            "candidate_txs",
            "grid_definitions"
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def compute(self, ctx):
        candidate_txs = ctx["candidate_txs"]
        grid_definitions = ctx["grid_definitions"]
        local2grid = grid_definitions["local2grid"]
        width, height = grid_definitions["local_coord_size"]

        # convert all coordinates to grid coordinates
        # idealizing the api to use rasterio, how would it work?
        normalized_heights = (candidate_txs["height"] - self.min_h) / (self.max_h - self.min_h)

        normalized_heights = np.clip(normalized_heights, 0.0, 1.0)

        normalized_heights_u8 = (normalized_heights * 255).astype(np.uint8)

        raster = np.zeros((height, width), dtype=np.uint8)
        rasterize(
            [(geom, height) for geom, height in zip(candidate_txs.geometry, normalized_heights_u8)],
            out=raster,
            transform=local2grid
        )

        grid_bboxes = grid_definitions["bboxes"]
        save_folder = self.cache_dirs(self.cache_keys(ctx))[0]
        save_folder.mkdir(parents=True, exist_ok=True)

        grid_idxs_to_mask = []
        num_of_txs_per_grid = []
        accumulated_num_of_txs = 0
        with self._logger.tqdm_progress_bar(
            enumerate(grid_bboxes),
            total=len(grid_bboxes),
            desc="Saving tx height feature grids"
        ) as pbar:
            for idx, (bbox_min, bbox_max) in pbar:
                if idx in grid_definitions["dangerously_mutable__masked_indices"]:
                    num_of_txs_per_grid.append(0)
                    continue

                grid_raster = raster[
                    bbox_min[1]:bbox_max[1],
                    bbox_min[0]:bbox_max[0],
                ]

                specific_tx_raster = np.zeros_like(grid_raster, dtype=np.uint8)
                txs_y, txs_x = np.where(grid_raster > 0)
                num_of_txs_per_grid.append(len(txs_x))

                if self.mask_when_num_of_tx_less_than is not None:
                    total_tx_cells = np.sum(grid_raster > 0)
                    if total_tx_cells < self.mask_when_num_of_tx_less_than:
                        grid_idxs_to_mask.append(idx)
                        grid_definitions["dangerously_mutable__masked_indices"].add(idx)
                        continue

                accumulated_num_of_txs += len(txs_x)

                for tx_k in range(len(txs_x)):
                    specific_tx_raster[txs_y[tx_k], txs_x[tx_k]] = grid_raster[txs_y[tx_k], txs_x[tx_k]]
                    cv2.imwrite(
                        str((save_folder /f"{idx}_{tx_k}.png").resolve()),
                        specific_tx_raster,
                    )
                    specific_tx_raster[txs_y[tx_k], txs_x[tx_k]] = 0
        folder_in_dataset = save_folder.resolve().relative_to(self._cache_dir.resolve())

        self._logger.debug(
            f"Masked {len(grid_idxs_to_mask)} grids out of {len(grid_bboxes)} total grids"
        )
        self._logger.info(
            f"Saved tx height feature grids to folder {str(save_folder.resolve())}"
        )

        return {
            self.feature_name: {
                "folder": save_folder,
                "masked_grid_indices": grid_idxs_to_mask,
                "feature_files_for_grid": [
                    [
                        folder_in_dataset / f"{idx}_{tx_k}.png"
                        for tx_k in range(num_of_txs_per_grid[idx])
                    ] for idx in range(len(grid_bboxes))
                ],
            },
        }

class FeatureTxHeightRasterOnGrids(FeatureAcquisitionStep):
    def __init__(
        self,
        dataset_dir: Path,
        min_h: float,
        max_h: float,
        feature_name: str = "tx_height",
        mask_when_num_of_tx_less_than: float = 1,
    ):
        super().__init__(dataset_dir, feature_name)
        self.min_h = min_h
        self.max_h = max_h
        self.mask_when_num_of_tx_less_than = mask_when_num_of_tx_less_than

    def cache_keys(self, ctx):
        return [self.feature_name + f"_{self.min_h}_{self.max_h}/"]

    def parse_ctx(self, raw_ctx) -> typing.Any:
        for required_key in [
            "candidate_txs",
            "grid_definitions"
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def compute(self, ctx):
        candidate_txs = ctx["candidate_txs"]
        grid_definitions = ctx["grid_definitions"]
        local2grid = grid_definitions["local2grid"]
        width, height = grid_definitions["local_coord_size"]

        # convert all coordinates to grid coordinates
        # idealizing the api to use rasterio, how would it work?
        normalized_heights = (candidate_txs["height"] - self.min_h) / (self.max_h - self.min_h)

        normalized_heights = np.clip(normalized_heights, 0.0, 1.0)

        normalized_heights_u8 = (normalized_heights * 255).astype(np.uint8)

        raster = np.zeros((height, width), dtype=np.uint8)
        rasterize(
            [(geom, height) for geom, height in zip(candidate_txs.geometry, normalized_heights_u8)],
            out=raster,
            transform=local2grid
        )

        grid_bboxes = grid_definitions["bboxes"]
        save_folder = self.cache_dirs(self.cache_keys(ctx))[0]
        save_folder.mkdir(parents=True, exist_ok=True)

        grid_idxs_to_mask = []
        num_of_txs_per_grid = []
        accumulated_num_of_txs = 0
        with self._logger.tqdm_progress_bar(
            enumerate(grid_bboxes),
            total=len(grid_bboxes),
            desc="Saving tx height feature grids"
        ) as pbar:
            for idx, (bbox_min, bbox_max) in pbar:
                if idx in grid_definitions["dangerously_mutable__masked_indices"]:
                    num_of_txs_per_grid.append(0)
                    continue

                grid_raster = raster[
                    bbox_min[1]:bbox_max[1],
                    bbox_min[0]:bbox_max[0],
                ]

                specific_tx_raster = np.zeros_like(grid_raster, dtype=np.uint8)
                txs_y, txs_x = np.where(grid_raster > 0)
                num_of_txs_per_grid.append(len(txs_x))

                if self.mask_when_num_of_tx_less_than is not None:
                    total_tx_cells = np.sum(grid_raster > 0)
                    if total_tx_cells < self.mask_when_num_of_tx_less_than:
                        grid_idxs_to_mask.append(idx)
                        grid_definitions["dangerously_mutable__masked_indices"].add(idx)
                        continue

                accumulated_num_of_txs += len(txs_x)

                for tx_k in range(len(txs_x)):
                    specific_tx_raster[txs_y[tx_k], txs_x[tx_k]] = grid_raster[txs_y[tx_k], txs_x[tx_k]]
                    cv2.imwrite(
                        str((save_folder /f"{idx}_{tx_k}.png").resolve()),
                        specific_tx_raster,
                    )
                    specific_tx_raster[txs_y[tx_k], txs_x[tx_k]] = 0
        folder_in_dataset = save_folder.resolve().relative_to(self._cache_dir.resolve())

        self._logger.debug(
            f"Masked {len(grid_idxs_to_mask)} grids out of {len(grid_bboxes)} total grids"
        )
        self._logger.info(
            f"Saved tx height feature grids to folder {str(save_folder.resolve())}"
        )

        return {
            self.feature_name: {
                "folder": save_folder,
                "masked_grid_indices": grid_idxs_to_mask,
                "feature_files_for_grid": [
                    [
                        folder_in_dataset / f"{idx}_{tx_k}.png"
                        for tx_k in range(num_of_txs_per_grid[idx])
                    ] for idx in range(len(grid_bboxes))
                ],
            },
        }

class FeatureTxFreeSpaceGainRasterOnGrids(FeatureAcquisitionStep):
    def __init__(
        self,
        dataset_dir: Path,
        frequency: float,
        rx_h: float,
        min_gain: float,
        max_gain: float,
        feature_name: str = "tx_fspl",
    ):
        super().__init__(dataset_dir, feature_name)
        self.frequency = frequency
        self.rx_h = rx_h
        self.min_gain = min_gain
        self.max_gain = max_gain
        if self.min_gain >= self.max_gain:
            raise ValueError("min_gain must be less than max_gain")
        if self.max_gain > 0:
            raise ValueError("max_gain must be less than or equal to 0")

    def cache_keys(self, ctx):
        return [self.feature_name + f"_{self.frequency}_{self.min_gain}_{self.max_gain}_{self.rx_h}/"]

    def parse_ctx(self, raw_ctx) -> typing.Any:
        for required_key in [
            "candidate_txs",
            "grid_definitions"
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def compute(self, ctx):
        candidate_txs = ctx["candidate_txs"]
        grid_definitions = ctx["grid_definitions"]
        local2grid = grid_definitions["local2grid"]
        width, height = grid_definitions["local_coord_size"]

        heights = np.clip(candidate_txs["height"], a_min=0, a_max=None)

        tx_raster = np.zeros((height, width), dtype=float)
        rasterize(
            [(geom, height) for geom, height in zip(candidate_txs.geometry, heights)],
            out=tx_raster,
            transform=local2grid
        )

        grid_bboxes = grid_definitions["bboxes"]
        save_folder = self.cache_dirs(self.cache_keys(ctx))[0]
        save_folder.mkdir(parents=True, exist_ok=True)

        num_of_txs_per_grid = []
        accumulated_num_of_txs = 0
        with self._logger.tqdm_progress_bar(
            enumerate(grid_bboxes),
            total=len(grid_bboxes),
            desc="Saving tx free space gain feature grids"
        ) as pbar:
            for idx, (bbox_min, bbox_max) in pbar:
                if idx in grid_definitions["dangerously_mutable__masked_indices"]:
                    num_of_txs_per_grid.append(0)
                    continue
                grid_len_x = (bbox_max[0] - bbox_min[0])
                grid_len_y = (bbox_max[1] - bbox_min[1])
                tx_at_raster = tx_raster[
                    bbox_min[1]:bbox_max[1],
                    bbox_min[0]:bbox_max[0],
                ]

                txs_y, txs_x = np.where(tx_at_raster > 0)
                num_of_txs_per_grid.append(len(txs_x))

                accumulated_num_of_txs += len(txs_x)

                for tx_k in range(len(txs_x)):
                    tx_h = tx_at_raster[txs_y[tx_k], txs_x[tx_k]]
                    tx_x, tx_y = txs_x[tx_k], txs_y[tx_k]
                    dists_from_tx = np.sqrt(
                        (tx_h - self.rx_h)**2 +
                        (np.arange(grid_len_x) - tx_x) ** 2 +
                        (np.arange(grid_len_y)[:, None] - tx_y) ** 2
                    )
                    fspl_raster = 20 * np.log10(dists_from_tx) + 20 * np.log10(4 * np.pi * self.frequency / 3e8)
                    free_space_gain_grayscale_raster = np.clip(
                        (-fspl_raster - self.min_gain) / (self.max_gain - self.min_gain) * 255,
                        0, 255
                    ).astype(np.uint8)
                    cv2.imwrite(
                        str((save_folder / f"{idx}_{tx_k}.png").resolve()),
                        free_space_gain_grayscale_raster,
                        )

        folder_in_dataset = save_folder.resolve().relative_to(self._cache_dir.resolve())

        self._logger.info(
            f"Saved tx free space gain feature grids to folder {str(save_folder.resolve())}"
        )

        return {
            self.feature_name: {
                "folder": save_folder,
                "feature_files_for_grid": [
                    [
                        folder_in_dataset / f"{idx}_{tx_k}.png"
                        for tx_k in range(num_of_txs_per_grid[idx])
                    ] for idx in range(len(grid_bboxes))
                ],
            },
        }


