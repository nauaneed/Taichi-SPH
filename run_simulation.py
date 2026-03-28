import os
import argparse
import importlib
import taichi as ti
import numpy as np
h5py = importlib.import_module("h5py") if importlib.util.find_spec("h5py") is not None else None
from SPH.utils import SimConfig
from SPH.containers import DFSPHContainer, WCSPHContainer, PCISPHContainer, PBFContainer, IISPHContainer
from SPH.fluid_solvers import DFSPHSolver, WCSPHSolver, PCISPHSolver, PBFSolver, IISPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.8)


def export_hdf5_xdmf(frame_dir, frame_idx, sim_time, fluid_data):
    if h5py is None:
        raise ImportError("h5py is required for HDF5 export. Install it via `pip install h5py`.")

    h5_path = os.path.join(frame_dir, "particles.h5")
    xdmf_path = os.path.join(frame_dir, "particles.xdmf")

    with h5py.File(h5_path, "w") as h5f:
        h5f.attrs["frame"] = frame_idx
        h5f.attrs["time"] = sim_time
        for body_id, obj_data in fluid_data.items():
            np_pos = obj_data["position"]
            np_vel = obj_data["velocity"]
            np_acc = obj_data["acceleration"]
            grp = h5f.create_group(f"object_{body_id}")
            grp.attrs["object_id"] = int(body_id)
            grp.attrs["num_particles"] = int(np_pos.shape[0])
            grp.attrs["time"] = sim_time
            grp.create_dataset("x", data=np_pos[:, 0].astype(np.float32), compression="gzip")
            grp.create_dataset("y", data=np_pos[:, 1].astype(np.float32), compression="gzip")
            grp.create_dataset("z", data=np_pos[:, 2].astype(np.float32), compression="gzip")
            grp.create_dataset("position", data=np_pos.astype(np.float32), compression="gzip")
            grp.create_dataset("vx", data=np_vel[:, 0].astype(np.float32), compression="gzip")
            grp.create_dataset("vy", data=np_vel[:, 1].astype(np.float32), compression="gzip")
            grp.create_dataset("vz", data=np_vel[:, 2].astype(np.float32), compression="gzip")
            grp.create_dataset("velocity", data=np_vel.astype(np.float32), compression="gzip")
            grp.create_dataset("ax", data=np_acc[:, 0].astype(np.float32), compression="gzip")
            grp.create_dataset("ay", data=np_acc[:, 1].astype(np.float32), compression="gzip")
            grp.create_dataset("az", data=np_acc[:, 2].astype(np.float32), compression="gzip")
            grp.create_dataset("acceleration", data=np_acc.astype(np.float32), compression="gzip")
            grp.create_dataset("rho", data=obj_data["rho"].astype(np.float32), compression="gzip")
            grp.create_dataset("p", data=obj_data["p"].astype(np.float32), compression="gzip")
            grp.create_dataset("temperature", data=obj_data["temperature"].astype(np.float32), compression="gzip")
            grp.create_dataset("viscosity", data=obj_data["viscosity"].astype(np.float32), compression="gzip")
            grp.create_dataset("m", data=obj_data["m"].astype(np.float32), compression="gzip")
            grp.create_dataset("time", data=np.full(np_pos.shape[0], sim_time, dtype=np.float32), compression="gzip")
            grp.create_dataset("rest_volume", data=obj_data["rest_volume"].astype(np.float32), compression="gzip")
            grp.create_dataset("material", data=obj_data["material"].astype(np.int32), compression="gzip")
            grp.create_dataset("is_dynamic", data=obj_data["is_dynamic"].astype(np.int32), compression="gzip")

    xdmf_lines = [
        "<?xml version=\"1.0\" ?>",
        "<Xdmf Version=\"3.0\">",
        "  <Domain>",
        f"    <Grid Name=\"frame_{frame_idx:06d}\" GridType=\"Collection\" CollectionType=\"Spatial\">",
        f"      <Time Value=\"{sim_time:.9f}\"/>",
    ]

    for body_id, obj_data in fluid_data.items():
        np_pos = obj_data["position"]
        n_particles = np_pos.shape[0]
        xdmf_lines.extend([
            f"      <Grid Name=\"fluid_object_{body_id}\" GridType=\"Uniform\">",
            f"        <Topology TopologyType=\"Polyvertex\" NumberOfElements=\"{n_particles}\"/>",
            "        <Geometry GeometryType=\"XYZ\">",
            f"          <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/position",
            "          </DataItem>",
            "        </Geometry>",
            "        <Attribute Name=\"rho\" AttributeType=\"Scalar\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/rho",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"p\" AttributeType=\"Scalar\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/p",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"temperature\" AttributeType=\"Scalar\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/temperature",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"viscosity\" AttributeType=\"Scalar\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/viscosity",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"m\" AttributeType=\"Scalar\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/m",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"velocity\" AttributeType=\"Vector\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/velocity",
            "          </DataItem>",
            "        </Attribute>",
            "        <Attribute Name=\"acceleration\" AttributeType=\"Vector\" Center=\"Node\">",
            f"          <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
            f"            particles.h5:/object_{body_id}/acceleration",
            "          </DataItem>",
            "        </Attribute>",
            "      </Grid>",
        ])

    xdmf_lines.extend([
        "    </Grid>",
        "  </Domain>",
        "</Xdmf>",
        "",
    ])

    with open(xdmf_path, "w") as xdmf_file:
        xdmf_file.write("\n".join(xdmf_lines))

#! due to code legacy, please use domain_start = [0, 0, 0]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    parser.add_argument('--export_hdf5_xdmf',
                        action='store_true',
                        help='force enable HDF5 + XDMF export')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    output_frames = config.get_cfg("exportFrame")

    fps = config.get_cfg("fps")
    if fps == None:
        fps = 60

    frame_time = 1.0 / fps

    output_interval = int(frame_time / config.get_cfg("timeStepSize"))

    total_time = config.get_cfg("totalTime")
    if total_time == None:
        total_time = 10.0

    total_rounds = int(total_time / config.get_cfg("timeStepSize"))
    
    if config.get_cfg("outputInterval"):
        output_interval = config.get_cfg("outputInterval")

    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    output_hdf5_xdmf = False if config.get_cfg("exportHdf5Xdmf") is None else bool(config.get_cfg("exportHdf5Xdmf"))
    output_hdf5_xdmf = output_hdf5_xdmf or args.export_hdf5_xdmf

    os.makedirs(f"{scene_name}_output", exist_ok=True)

    simulation_method = config.get_cfg("simulationMethod")
    if simulation_method == "dfsph":
        container = DFSPHContainer(config, GGUI=True)
        solver = DFSPHSolver(container)
    elif simulation_method == "wcsph":
        container = WCSPHContainer(config, GGUI=True)
        solver = WCSPHSolver(container)
    elif simulation_method == "pcisph":
        container = PCISPHContainer(config, GGUI=True)
        solver = PCISPHSolver(container)
    elif simulation_method == "iisph":
        container = IISPHContainer(config, GGUI=True)
        solver = IISPHSolver(container)
    elif simulation_method == "pbf":
        container = PBFContainer(config, GGUI=True)
        solver = PBFSolver(container)
    else:
        raise NotImplementedError(f"Simulation method {simulation_method} not implemented")

    print(f"Simulation method: {simulation_method}")

    solver.prepare()


    window = ti.ui.Window('SPH', (1024, 1024), show_window = False, vsync=False)

    scene = ti.ui.Scene()
    # feel free to adjust the position of the camera as needed
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    # Draw the lines for domain
    domain_end = config.get_cfg("domainEnd")
    dim = len(domain_end)
    if len(domain_end) == 3:
        x_max, y_max, z_max = domain_end
        box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
        box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
        box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
        box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
        box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

        box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
        box_anchors[5] = ti.Vector([0.0, y_max, z_max])
        box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
        box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))

    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    cnt = 0
    cnt_ply = 0

    while window.running:
        solver.step()
        container.copy_to_vis_buffer(invisible_objects=invisible_objects, dim=dim)
        if container.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(container.x_vis_buffer, radius=container.dx / 80.0, color=particle_color)
        elif container.dim == 3:
            scene.set_camera(camera)

            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(container.x_vis_buffer, radius=container.dx, per_vertex_color=container.color_vis_buffer)

            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
            canvas.scene(scene)
    
        if output_frames:
            if cnt % output_interval == 0:
                os.makedirs(f"{scene_name}_output/{cnt:06}", exist_ok=True)
                window.save_image(f"{scene_name}_output/{cnt:06}/raw_view.png")
        
        if cnt % output_interval == 0:
            frame_dir = f"{scene_name}_output/{cnt:06}"
            if output_ply:
                os.makedirs(frame_dir, exist_ok=True)
                for f_body_id in container.object_id_fluid_body:
                    obj_data = container.dump(obj_id=f_body_id)
                    np_pos = obj_data["position"]
                    writer = ti.tools.PLYWriter(num_vertices=container.object_collection[f_body_id]["particleNum"])
                    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                    writer.export_ascii(f"{frame_dir}/particle_object_{f_body_id}.ply")
            if output_obj:
                os.makedirs(frame_dir, exist_ok=True)
                for r_body_id in container.object_id_rigid_body:
                    with open(f"{frame_dir}/mesh_object_{r_body_id}.obj", "w") as f:
                        e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)
            if output_hdf5_xdmf:
                os.makedirs(frame_dir, exist_ok=True)
                fluid_data = {}
                for f_body_id in container.object_id_fluid_body:
                    obj_data = container.dump(obj_id=f_body_id)
                    fluid_data[f_body_id] = obj_data
                sim_time = cnt * config.get_cfg("timeStepSize")
                export_hdf5_xdmf(frame_dir, cnt, sim_time, fluid_data)

        cnt += 1

        if cnt >= total_rounds:
            break

    print(f"Simulation Finished")