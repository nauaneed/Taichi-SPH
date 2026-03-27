# copy from https://github.com/erizmr/SPH_Taichi/blob/master/config_builder.py
import json


def _strip_json_comments(text):
    in_string = False
    quote_char = ""
    escaped = False
    i = 0
    out = []
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                in_string = False
            i += 1
            continue

        if ch == '"' or ch == "'":
            in_string = True
            quote_char = ch
            out.append(ch)
            i += 1
            continue

        # line comment: //...
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue

        # block comment: /* ... */
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _strip_trailing_commas(text):
    in_string = False
    quote_char = ""
    escaped = False
    i = 0
    out = []
    n = len(text)

    while i < n:
        ch = text[i]

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                in_string = False
            i += 1
            continue

        if ch == '"' or ch == "'":
            in_string = True
            quote_char = ch
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "]}":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _load_scene_config(scene_file_path):
    with open(scene_file_path, "r") as f:
        raw = f.read()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _strip_trailing_commas(_strip_json_comments(raw))
        return json.loads(cleaned)


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        self.config = _load_scene_config(scene_file_path)
        print(self.config)
    
    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["Configuration"]
        if name not in self.config["Configuration"]:
            if enforce_exist:
                assert name in self.config["Configuration"]
            else:
                return None
        return self.config["Configuration"][name]
    
    def get_rigid_bodies(self):
        if "RigidBodies" in self.config:
            return self.config["RigidBodies"]
        else:
            return []
    
    def get_rigid_blocks(self):
        if "RigidBlocks" in self.config:
            return self.config["RigidBlocks"]
        else:
            return []

    def get_fluid_bodies(self):
        if "FluidBodies" in self.config:
            return self.config["FluidBodies"]
        else:
            return []
    
    def get_fluid_blocks(self):
        if "FluidBlocks" in self.config:
            return self.config["FluidBlocks"]
        else:
            return []
