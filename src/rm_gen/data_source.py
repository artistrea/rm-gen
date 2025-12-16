"""
This module contains data source steps for downloading and processing
building footprints from OpenStreetMap (OSM) data, as well as steps
for assigning building heights.

Its purpose is to fetch geospatial data for the RM-Gen pipeline.
"""

import hashlib
import shutil
import subprocess
import typing
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import osmnx as ox
import pandas as pd
import requests
import shapely.wkb as wkblib

from .core import Step
from .log_utils import _LoggerWithTQDM


def _run_command(command, logger: _LoggerWithTQDM = None):
    """
    Runs a command using subprocess.run and logs the command and any errors.

    Parameters
    ----------
    command: list or str
        The command to run as a list of arguments or a single string.
    logger: _LoggerWithTQDM, optional
        Logger to use for logging. Defaults to None.

    Raises
    ------
        Exception: Reraises any exception that occurs during command execution.
    """
    try:
        if isinstance(command, list):
            command_string = " ".join(command)
        else:
            command_string = command

        if logger is not None:
            logger.debug("RUNNING: " + command_string)
        subprocess.run(
            command,
            check=True
        )
    # pylint: disable=broad-except
    except Exception as e:
        if logger is not None:
            logger.error("COMMAND FAILED: " + command_string)
            logger.error(str(e))
        raise e


def _download_file(
    url: str,
    dst: str,
    chunk_size=1024*1024,
    logger: _LoggerWithTQDM = None,
):
    """
    Downloads a file from a URL to a destination path with progress bar.

    Parameters
    ----------
    url: str
        The URL to download the file from.
    dst: str
        The destination path to save the downloaded file.
    chunk_size: int, optional
        The size of each chunk to read from the response. Defaults to 1 MB.
    logger: _LoggerWithTQDM, optional
        Logger to use for logging progress. Defaults to None.

    Returns
    -------
        str: The path to the downloaded file.

    Raises
    ------
        Exception: Reraises any exception that occurs during download.

    Notes
    -----
        If the download is interrupted or fails, the partially downloaded file
        is deleted.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    try:
        print(f"Downloading {url} to {dst}")
        with open(dst, "wb") as f, logger.tqdm_progress_bar(
            None,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    progress.update(len(chunk))
                    f.write(chunk)
    except KeyboardInterrupt as e:
        if Path(dst).exists():
            Path(dst).unlink()
        raise e
    # pylint: disable=broad-except
    except Exception as e:
        if Path(dst).exists():
            Path(dst).unlink()
        raise e

    return dst

class _OsmiumBuildingFootprintHandler(osmium.SimpleHandler):
    """
    Osmium handler to extract building footprints with height and levels
    from OSM data.
    """
    def __init__(self):
        super().__init__()
        # pylint: disable=c-extension-no-member
        self.wkbfab = osmium.geom.WKBFactory()
        self.rows = []

    def area(self, a):
        """
        Handle an OSM area object to extract building footprint geometry
        and attributes.
        """
        try:
            wkb = self.wkbfab.create_multipolygon(a)
        # pylint: disable=broad-except
        except Exception:
            return
        geom_shape = wkblib.loads(wkb, hex=True)

        tags = a.tags
        height = tags.get("height")
        levels = tags.get("building:levels")

        self.rows.append({
            "geometry": geom_shape,
            "height": height,
            "building:levels": levels,
        })


class BuildingFootprintFromDownload(Step):
    """
    Downloads OSM data from a given link, applies a bounding box,
    filters for building footprints, and adds to context a
    geopandas.GeoDataFrame with buildings footprint (w/ height when possible).

    Notes
    -----
        Requires `osmium` command line tool to be installed.

    Example
    -------
        >>> step = BuildingFootprintFromDownload(
        ...     download_link="download.geofabrik.de/south-america/brazil/\
sudeste-latest.osm.pbf",
        ...     lonlat_bounding_box=((-43.2319, -22.99), (-43.190, -22.96))
        ... )
        >>> result = step.run()
    """
    def __init__(
        self,
        download_link: str,
        lonlat_bounding_box: typing.Tuple[
            typing.Tuple[float, float],
            typing.Tuple[float, float]
        ],
    ):
        """
        Parameters
        ----------
        download_link: str
            The URL to download the OSM data from.
        lonlat_bounding_box: tuple of tuple of float
            The bounding box to extract from the OSM data in
            ((min_lon, min_lat), (max_lon, max_lat)) format.
        """
        super().__init__()
        self.download_link = download_link
        self.lonlat_bounding_box = lonlat_bounding_box
        if shutil.which("osmium") is None:
            # what error to raise here?
            raise RuntimeError(
                "You need to install osmium to be able to run this step"
            )

    def cache_keys(self, ctx):
        download_hash = hashlib.md5(
            self.download_link.encode()
        ).hexdigest()

        bbox = self.lonlat_bounding_box
        bbox_hash = hashlib.md5(
            (str(bbox) + "---" + self.download_link).encode()
        ).hexdigest()

        return [
            f"downloads/{download_hash}.pbf",
            f"processed/bbox/{bbox_hash}.pbf",
            f"processed/building_fps/{bbox_hash}.pbf",
            f"processed/building_fps/{bbox_hash}.geojson",
        ]

    def load_cache(self, keys):
        _, _, _, fp = self.cache_dirs(keys)
        gdf = gpd.read_file(fp)

        self._logger.info(
            "Number of building footprints loaded " + str(len(gdf))
        )
        return {"building_footprints": gdf}

    def parse_ctx(self, raw_ctx):
        return raw_ctx

    def compute(self, ctx):
        """
        Downloads OSM data, applies bounding box, filters for buildings,
        and extracts building footprints into a GeoDataFrame.

        Parameters
        ----------
        ctx: dict
            The context dictionary (not used in this step).

        Returns
        -------
        dict
            A dictionary with a single key 'building_footprints'
            containing a geopandas.GeoDataFrame of building footprints.
        """
        keys = self.cache_keys(ctx)
        download_path, bboxed_path, buildings_path, fp_path =\
            self.cache_dirs(keys)

        if not download_path.exists():
            self._logger.debug("Download OSM file")
            _download_file(
                self.download_link,
                download_path,
                1024 * 1024,
                self._logger
            )
        else:
            self._logger.debug("Using cached downloaded OSM file")

        if not bboxed_path.exists():
            self._logger.debug("Applying bbox on OSM data")
            bbox = self.lonlat_bounding_box
            bbox_str = ",".join([
                str(bbox[0][0]), str(bbox[0][1]),
                str(bbox[1][0]), str(bbox[1][1])
            ])
            _run_command(["osmium", "extract",
                "-b", bbox_str,
                str(download_path),
                "-o", str(bboxed_path),
                # "--overwrite"
            ])
        else:
            self._logger.debug("Using cached bbox applied OSM data")

        if not buildings_path.exists():
            self._logger.debug(
                "Applying filter to get only buildings from OSM data"
            )
            _run_command(["osmium", "tags-filter",
                str(bboxed_path),
                "nwr", "building",
                "-o", str(buildings_path),
                # "--overwrite"
            ])
        else:
            self._logger.debug(
                "Using cached filter with only buildings from OSM data"
            )

        if not fp_path.exists():
            self._logger.debug(
                "Filtering only wanted properties and creating final GeoJSON"
            )
            handler = _OsmiumBuildingFootprintHandler()
            handler.apply_file(str(buildings_path.resolve()))

            gdf = gpd.GeoDataFrame(handler.rows, crs="EPSG:4326")

            if gdf.empty:
                raise ValueError("Place does not contain any buildings")

            # get rid of bad labeling people do
            # and only accept geometry with area as building
            gdf = gdf.loc[
                gdf.geometry.notnull() &
                gdf.geometry.type.isin(['Polygon','MultiPolygon'])
            ]
            gdf.to_file(fp_path, driver="GeoJSON")
        else:
            self._logger.debug("Using cached final GeoJSON")

        fps = gpd.read_file(fp_path)
        self._logger.info(
            "Number of building footprints loaded " + str(len(fps))
        )

        return {"building_footprints": fps}


class BuildingFootprintFromOSMPlace(Step):
    """
    Downloads building footprints from OSM, using OverPass API,
    for a given place query and adds to context a geopandas.GeoDataFrame
    with buildings footprint (w/ height when possible).

    Example
    -------
        >>> step = BuildingFootprintFromOSMPlace(
        ...     place_query="Plano Piloto, Brasília, DF, Brasil",
        ... )
        >>> result = step.run()
    """
    def __init__(
        self,
        place_query: str,
    ):
        """
        Parameters
        ----------
        place_query: str
            The place query to search for buildings in OSM.
        """
        super().__init__()
        self.place_query = place_query

    def cache_keys(self, ctx):
        place_query = self.place_query
        # place_query_hash = hashlib.md5(
        #     self.place_query.encode()
        # ).hexdigest()

        return [
            f"{place_query}.geojson",
        ]

    def load_cache(self, keys):
        fp = self.cache_dirs(keys)[0]
        gdf = gpd.read_file(fp)
        self._logger.info(
            "Number of building footprints loaded " + str(len(gdf))
        )

        return {"building_footprints": gdf}

    def parse_ctx(self, raw_ctx):
        return raw_ctx

    def compute(self, ctx):
        keys = self.cache_keys(ctx)
        fp_path = self.cache_dirs(keys)[0]

        if not fp_path.exists():
            tags = {'building': True}
            buildings = ox.features_from_place(self.place_query, tags=tags)

            if buildings.empty:
                raise ValueError("Place does not contain any buildings")

            # get rid of bad labeling people do
            # and only accept geometry with area as building
            buildings = buildings.loc[
                buildings.geometry.notnull() &
                buildings.geometry.type.isin(['Polygon','MultiPolygon'])
            ]
            buildings.to_file(fp_path, driver="GeoJSON")

        fps = gpd.read_file(fp_path)

        self._logger.info(
            "Number of building footprints loaded " + str(len(fps))
        )

        return {"building_footprints": fps}


class RandomBuildingHeight(Step):
    """Decide on building height for posterior extrusion
    """
    def __init__(
        self,
        min_height=6.6,
        max_height=19.8,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed=32)
        self.height_per_level = 3.0
        self.min_height = min_height
        self.max_height = max_height

    def parse_ctx(self, raw_ctx):
        if "building_footprints" not in raw_ctx:
            raise ValueError("Context must contain 'building_footprints'.")
        return raw_ctx

    def compute(self, ctx):
        building_fps = ctx["building_footprints"]

        height = building_fps["height"].to_numpy()
        height = self.rng.uniform(
            self.min_height,
            self.max_height,
            size=height.shape
        )

        return {"building_heights": height}

class NaiveBuildingHeight(Step):
    """Decide on building height for posterior extrusion
    """
    def __init__(
        self,
        default_height=12.0
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed=32)
        self.height_per_level = 3.0
        self.default_height = default_height

    def parse_ctx(self, raw_ctx):
        if "building_footprints" not in raw_ctx:
            raise ValueError("Context must contain 'building_footprints'.")
        return raw_ctx

    def compute(self, ctx):
        building_fps = ctx["building_footprints"]

        height = pd.to_numeric(building_fps["height"], errors="coerce")
        levels = self.height_per_level * pd.to_numeric(
            building_fps["building:levels"], errors="coerce"
        )

        h = height.fillna(levels)

        nans = h.isna()

        h[nans] = self.default_height

        height_m = h.to_numpy()
        return {"building_heights": height_m}
