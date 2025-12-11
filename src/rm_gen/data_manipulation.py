import hashlib
import shutil
import typing
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely as shp
import trimesh
from affine import Affine

from .core import Step


class CreateFullMitsubaScene(Step):
    def __init__(
        self,
        floor_mat="itu_medium_dry_ground",
        building_mat="itu_concrete",
        clean_cached_scene=False,
    ):
        super().__init__()
        self.floor_mat = floor_mat
        self.building_mat = building_mat
        self.clean_cached_scene = clean_cached_scene
        self.materials_used = set([
            self.floor_mat,
            self.building_mat,
        ])

    # TODO: check if all meshes inside scene actually exist before
    # loading from cache
    def cache_keys(self, ctx):
        fps = str(ctx["building_footprints"])
        hs = str(ctx["building_heights"])

        scene_hash = hashlib.md5(
            "---".join([fps, hs, self.floor_mat, self.building_mat]).encode()
        ).hexdigest()

        return [
            f"{scene_hash}/meshes/",
            f"{scene_hash}/scene.xml",
        ]

    def load_cache(self, keys):
        file_path = self.cache_dirs(keys)[-1]
        return {"mitsuba_scene_path": file_path.resolve()}

    def parse_ctx(self, raw_ctx):
        for required_key in [
            "local_coord_sys_attributes",
            "local_building_footprints",
            "building_heights",
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")
        return raw_ctx

    def clean_polygon(self, geom):
        """Returns a valid polygon or multipolygon."""
        geom = shp.make_valid(geom)
        if geom.is_empty:
            return None
        # Fix self-intersections and invalid geometries
        geom = geom.buffer(0)
        if geom.is_empty:
            return None
        if geom.geom_type == "MultiPolygon":
            # Keep only the largest component (optional)
            geom = max(geom.geoms, key=lambda g: g.area)
        return geom

    def get_building_ring(self, building_polygon: shp.Polygon):
        building_polygon = self.clean_polygon(building_polygon)
        exterior_coords = building_polygon.exterior.coords
        oriented_coords = list(exterior_coords)
        points = [(coord[0], coord[1]) for coord in oriented_coords]
        return points

    def add_buildings_to_scene(
        self, scene, actual_meshes_dir, xml_saved_meshes_dir, material_type,
        local_building_footprints, building_heights
    ):
        buildings_list = local_building_footprints.to_dict('records')
        skipped = 0

        with self._logger.tqdm_progress_bar(enumerate(buildings_list),
                                total=len(buildings_list),
                                desc="Creating building meshes") as building_iter:
            for idx, building in building_iter:
                # Convert building geometry to a shapely polygon
                building_poly = self.clean_polygon(
                    shp.geometry.shape(building['geometry'])
                )
                if building_poly is None:
                    skipped += 1
                    continue

                building_height = building_heights[idx]
                building_mesh = trimesh.creation.extrude_polygon(
                    building_poly,
                    height=building_height
                )

                # export mesh to PLY file
                building_mesh.export(str(actual_meshes_dir / f"building_{idx}.ply"))

                # Add shape elements for PLY files in the folder
                sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")

                # mesh path
                ET.SubElement(sionna_shape, "string", name="filename", value=str(xml_saved_meshes_dir / f"building_{idx}.ply"))

                # bsdf material
                ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")

                # TODO: check face_normals and its edge cases
                ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")

        self._logger.info(f"Skipping {skipped} buildings due to invalid geometry")

    def add_floor_to_scene(
        self,
        scene_xml: ET.Element,
        actual_meshes_dir: Path,
        xml_saved_meshes_dir: Path,
        material_type: str,
        minx: float, miny: float, maxx: float, maxy: float
    ):
        ground_polygon = shp.box(minx, miny, maxx, maxy, ccw=True)

        verts2d, faces2d = trimesh.creation.triangulate_polygon(ground_polygon)

        verts3d = np.column_stack([verts2d, np.full(len(verts2d), 0.0)])

        mesh3d = trimesh.Trimesh(vertices=verts3d, faces=faces2d)

        mesh3d.export(str(actual_meshes_dir / "ground.ply"))

        ground_shape = ET.SubElement(scene_xml, "shape", type="ply", id="mesh-ground")
        ET.SubElement(
            ground_shape, "string", name="filename",
            value=str(xml_saved_meshes_dir / "ground.ply")
        )
        ET.SubElement(ground_shape, "ref", id=material_type, name="bsdf")
        # TODO: check face_normals and its edge cases
        ET.SubElement(ground_shape, "boolean", name="face_normals",value="true")

    def prepare(self, ctx):
        if self.clean_cached_scene:
            self.clean_cache(ctx)

    def compute(self, ctx):
        keys = self.cache_keys(ctx)
        meshes_dir, scene_path = self.cache_dirs(keys)
        bounds = ctx["local_coord_sys_attributes"]["bounds"]

        local_building_footprints = ctx["local_building_footprints"]
        building_heights = ctx["building_heights"]

        # remove all existing meshes
        if meshes_dir.exists():
            shutil.rmtree(meshes_dir)
        # recreate them
        meshes_dir.mkdir(exist_ok=True)
        xml_saved_meshes_dir = meshes_dir.resolve().relative_to(scene_path.parent.resolve())

        scene = ET.Element("scene", version="2.1.0")

        minx, miny, maxx, maxy = bounds

        # Define materials
        for material_id in self.materials_used:
            bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=material_id)
            ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
            # ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

        self.add_floor_to_scene(
            scene, meshes_dir, xml_saved_meshes_dir, self.floor_mat, minx, miny, maxx, maxy
        )

        self.add_buildings_to_scene(
            scene, meshes_dir, xml_saved_meshes_dir, self.building_mat,
            local_building_footprints, building_heights
        )

        xml_string = ET.tostring(scene, encoding="utf-8")
        xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")

        with open(scene_path, "w", encoding="utf-8") as xml_file:
            xml_file.write(xml_pretty)

        return {"mitsuba_scene_path": scene_path.resolve()}

class BuildingFootprintToLocalCoords(Step):
    def __init__(
        self,
        origin_at="bottom_left",
        local_coords_bbox: typing.Optional[
            typing.Tuple[
                typing.Tuple[float, float],
                typing.Tuple[float, float]
            ]
        ] = None,
    ):
        if origin_at not in ["bottom_left", "center"]:
            raise ValueError("'origin_at' must be 'bottom_left' or 'center'")
        self.origin_at = origin_at
        self.local_coords_bbox = local_coords_bbox

        super().__init__()

    def parse_ctx(self, raw_ctx) -> typing.Any:
        if "building_footprints" not in raw_ctx:
            raise ValueError(
                "This step requires 'building_footprints' to be in context"
            )
        return raw_ctx

    def compute(self, ctx):
        fps = ctx["building_footprints"]
        if fps.crs != "EPSG:4326":
            raise ValueError(
                "Building footprints need to be in EPSG:4326 before conversion to local"
            )
        utm_crs = fps.estimate_utm_crs()
        utm_fps = fps.to_crs(utm_crs)
        minx, miny, maxx, maxy = utm_fps.total_bounds
        if self.origin_at == "bottom_left":
            utm_fps["geometry"] = utm_fps.translate(xoff=-minx, yoff=-miny)
            range_x = maxx - minx
            range_y = maxy - miny
            local_coord_sys_attributes ={
                    "center": (range_x/2, range_y/2),
                    "bounds": (0.0, 0.0, range_x, range_y),
                }
            local_building_footprints = utm_fps
        elif self.origin_at == "center":
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            range_x = maxx - minx
            range_y = maxy - miny
            utm_fps["geometry"] = utm_fps.translate(xoff=-center_x, yoff=-center_y)
            local_coord_sys_attributes ={
                "center": (0.0, 0.0),
                "bounds": (-range_x/2, -range_y/2, range_x/2, range_y/2),
            }
            local_building_footprints = utm_fps
        else:
            raise ValueError("Invalid 'origin_at' value")

        if self.local_coords_bbox is not None:
            (bbox_minx, bbox_miny), (bbox_maxx, bbox_maxy) = self.local_coords_bbox
            local_building_footprints = local_building_footprints.cx[
                bbox_minx:bbox_maxx,
                bbox_miny:bbox_maxy
            ]

        return {
            "local_coord_sys_attributes": local_coord_sys_attributes,
            "local_building_footprints": local_building_footprints,
        }

class GenerateCandidateTransmittersOnBuildings(Step):
    def __init__(
        self,
        height_above_building,
        min_building_height,
        step,
    ):
        self.height_above_building = height_above_building
        self.min_building_height = min_building_height
        self.step = step
        super().__init__()

    def cache_keys(self, ctx):
        fps = str(ctx["local_building_footprints"])
        hs = str(ctx["building_heights"])

        tx_hash = hashlib.md5(
            "---".join([
                fps,
                hs,
                str(self.height_above_building),
                str(self.min_building_height),
                str(self.step),
            ]).encode()
        ).hexdigest()

        return [
            f"candidate_txs/{tx_hash}.geojson",
        ]
    
    def load_cache(self, keys):
        fp = self.cache_dirs(keys)[0]
        gdf = gpd.read_file(fp)
        self._logger.info("Loaded " + str(len(gdf)) + " candidate transmitter locations from cache")
        return {"candidate_txs": gdf}

    def parse_ctx(self, raw_ctx) -> typing.Any:
        for required_key in [
            "local_building_footprints",
            "building_heights",
        ]:
            if required_key not in raw_ctx:
                raise ValueError(f"Context must contain '{required_key}'.")

        return raw_ctx

    def compute(self, ctx):
        cached_tx_dir = self.cache_dirs(self.cache_keys(ctx))[0]

        if cached_tx_dir.exists():
            return self.load_cache(self.cache_keys(ctx))

        local_building_fps = ctx["local_building_footprints"]
        building_heights = ctx["building_heights"]

        candidate_points = []
        tx_hs = []

        # prepare structure to query buildings and check whether a tx is inside
        # any
        polys = []
        heights = []

        for idx, row in local_building_fps.iterrows():
            poly = row["geometry"]
            if poly is None or poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                for subpoly in poly.geoms:
                    polys.append(subpoly)
                    heights.append(building_heights[idx])
            else:
                polys.append(poly)
                heights.append(building_heights[idx])

        tree = shp.STRtree(polys)

        step = self.step  # meters
        # consider intersection with other buildings within this distance
        intersection_distance = 1.0  # meters
        # height above another building to not be considered blocked
        clearance = 1.0  # meters

        min_building_height = self.min_building_height  # meters
        height_above_building = self.height_above_building  # meters

        with self._logger.tqdm_progress_bar(list(enumerate(polys)), desc="Generating candidate tx on buildings") as bar:
            for idx, poly in bar:
                building_height = heights[idx]
                if building_height < min_building_height:
                    continue
                tx_height = building_height + height_above_building
                line = poly.exterior
                length = line.length
                distances = np.arange(0, length - step, step)

                for d in distances:
                    pt2d = line.interpolate(d)

                    # check nearby polygons
                    nearby = tree.query(pt2d.buffer(intersection_distance))
                    covered = False

                    for j in nearby:
                        if j == idx:
                            continue

                        # and if height of that building is "too much"
                        if heights[j] >= tx_height - clearance:
                            covered = True
                            break

                    # adds 2d point
                    candidate_points.append(pt2d)
                    tx_hs.append(tx_height)

        gdf = gpd.GeoDataFrame(
            dict(height=[h for h in tx_hs]),
            geometry=candidate_points,
            crs=local_building_fps.crs,
        )

        keys = self.cache_keys(ctx)
        fp = self.cache_dirs(keys)[0]
        gdf.to_file(fp, driver="GeoJSON")

        # self._logger.info(gdf)
        self._logger.info("Generated " + str(len(gdf)) + " candidate transmitter locations")

        return {
            "candidate_txs": gdf,
        }

class GenerateScenariosInGrids(Step):
    def __init__(
        self,
        resolution: typing.Tuple[float, float],
        grid_len: typing.Tuple[int, int],
        grid_step: typing.Tuple[int, int],
    ):
        """Creates grids over the local coordinate system for scenario generation
        resolution: (x_res, y_res) in meters
        grid_len: (x_len, y_len) in number of cells
        grid_step: (x_step, y_step) in number of cells

        Windows over scenario from bottom left to top right, with given step and length
        256x256 cells with 128 step would create overlapping windows of 256x256 cells
        every 128 cells in x and y direction
        256x256 cells with 256 step would create non-overlapping windows of 256 cells
        """
        super().__init__()
        self.resolution = resolution
        self.grid_len = grid_len
        self.grid_step = grid_step

    def parse_ctx(self, raw_ctx) -> typing.Any:
        if "local_coord_sys_attributes" not in raw_ctx:
            raise ValueError(
                "This step requires 'local_coord_sys_attributes' to be in context"
            )
        return raw_ctx

    def compute(self, ctx):
        local_coord_sys_attributes = ctx["local_coord_sys_attributes"]
        scale = self.resolution

        local2grid = Affine.translation(
            -local_coord_sys_attributes["bounds"][0],
            -local_coord_sys_attributes["bounds"][1]
        ) * Affine.scale(1/scale[0], 1/scale[1])

        width, height = local2grid * (
            local_coord_sys_attributes["bounds"][2],
            local_coord_sys_attributes["bounds"][3],
        )
        width, height = int(width), int(height)
        step_x, step_y = self.grid_step
        len_x, len_y = self.grid_len

        self._logger.debug(
            f"Generating grids over grid coordinate system of size ({width}, {height}) "+
            f"with grid size ({len_x}, {len_y}) and step ({step_x}, {step_y})"
        )

        grid_bboxes = []
        for i, j in product(range(0, width, step_x), range(0, height, step_y)):
            if i + len_x >= width or j + len_y >= height:
                continue
            grid_bboxes.append((
                (i, j),
                (i + len_x, j + len_y),
            ))

        self._logger.info(f"Generated {len(grid_bboxes)} grids over local coordinate system")

        return {
            "grid_definitions": {
                "grid_size": (len_x, len_y),
                "local_coord_size": (width, height),
                "resolution": scale,
                "bboxes": grid_bboxes,
                "local2grid": local2grid,
                "grid2local": ~local2grid,
                # NOTE: you may mutate this structure later to add more info
                "dangerously_mutable__masked_indices": set()
            },
        }


