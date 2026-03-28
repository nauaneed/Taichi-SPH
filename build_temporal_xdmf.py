import argparse
import os
import re
import h5py


def _frame_key(name):
    return int(name) if name.isdigit() else name


def _sorted_frame_dirs(input_dir):
    frame_dirs = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        if not re.fullmatch(r"\d+", name):
            continue
        h5_path = os.path.join(path, "particles.h5")
        if os.path.isfile(h5_path):
            frame_dirs.append(name)
    frame_dirs.sort(key=_frame_key)
    return frame_dirs


def _sorted_object_groups(h5f):
    object_names = [name for name in h5f.keys() if re.fullmatch(r"object_\d+", name)]
    object_names.sort(key=lambda n: int(n.split("_")[1]))
    return object_names


def build_temporal_xdmf(input_dir, output_file):
    frame_dirs = _sorted_frame_dirs(input_dir)
    if not frame_dirs:
        raise RuntimeError(f"No frame folders with particles.h5 found in {input_dir}")

    lines = [
        "<?xml version=\"1.0\" ?>",
        "<Xdmf Version=\"3.0\">",
        "  <Domain>",
        "    <Grid Name=\"SPHTemporal\" GridType=\"Collection\" CollectionType=\"Temporal\">",
    ]

    for frame_name in frame_dirs:
        h5_rel = f"{frame_name}/particles.h5"
        h5_abs = os.path.join(input_dir, frame_name, "particles.h5")

        with h5py.File(h5_abs, "r") as h5f:
            frame_time = float(h5f.attrs.get("time", int(frame_name)))
            lines.extend([
                f"      <Grid Name=\"frame_{frame_name}\" GridType=\"Collection\" CollectionType=\"Spatial\">",
                f"        <Time Value=\"{frame_time:.9f}\"/>",
            ])

            for obj_name in _sorted_object_groups(h5f):
                grp = h5f[obj_name]
                object_id = int(obj_name.split("_")[1])
                if "num_particles" in grp.attrs:
                    n_particles = int(grp.attrs["num_particles"])
                elif "position" in grp:
                    n_particles = int(grp["position"].shape[0])
                elif "x" in grp:
                    n_particles = int(grp["x"].shape[0])
                else:
                    raise RuntimeError(f"Unable to infer particle count for {obj_name} in {h5_abs}")

                lines.extend([
                    f"        <Grid Name=\"fluid_object_{object_id}\" GridType=\"Uniform\">",
                    f"          <Topology TopologyType=\"Polyvertex\" NumberOfElements=\"{n_particles}\"/>",
                    "          <Geometry GeometryType=\"XYZ\">",
                    f"            <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/position",
                    "            </DataItem>",
                    "          </Geometry>",
                    "          <Attribute Name=\"rho\" AttributeType=\"Scalar\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/rho",
                    "            </DataItem>",
                    "          </Attribute>",
                    "          <Attribute Name=\"p\" AttributeType=\"Scalar\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/p",
                    "            </DataItem>",
                    "          </Attribute>",
                    "          <Attribute Name=\"temperature\" AttributeType=\"Scalar\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/temperature",
                    "            </DataItem>",
                    "          </Attribute>",
                    "          <Attribute Name=\"m\" AttributeType=\"Scalar\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/m",
                    "            </DataItem>",
                    "          </Attribute>",
                    "          <Attribute Name=\"velocity\" AttributeType=\"Vector\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/velocity",
                    "            </DataItem>",
                    "          </Attribute>",
                    "          <Attribute Name=\"acceleration\" AttributeType=\"Vector\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/acceleration",
                    "            </DataItem>",
                    "          </Attribute>",
                ])

                lines.extend([
                    "          <Attribute Name=\"viscosity\" AttributeType=\"Scalar\" Center=\"Node\">",
                    f"            <DataItem Dimensions=\"{n_particles} 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">",
                    f"              {h5_rel}:/{obj_name}/viscosity",
                    "            </DataItem>",
                    "          </Attribute>",
                ])

                lines.append("        </Grid>")

            lines.append("      </Grid>")

    lines.extend([
        "    </Grid>",
        "  </Domain>",
        "</Xdmf>",
        "",
    ])

    out_abs = os.path.join(input_dir, output_file)
    with open(out_abs, "w") as f:
        f.write("\n".join(lines))

    return out_abs, len(frame_dirs)


def main():
    parser = argparse.ArgumentParser(description="Build a single temporal XDMF index from per-frame particles.h5 files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Simulation output directory, e.g., ./final_scene4_output")
    parser.add_argument("--output_file", type=str, default="particles_temporal.xdmf", help="Name of output temporal XDMF file")
    args = parser.parse_args()

    out_path, frame_count = build_temporal_xdmf(args.input_dir, args.output_file)
    print(f"Wrote temporal XDMF: {out_path}")
    print(f"Frames indexed: {frame_count}")


if __name__ == "__main__":
    main()
