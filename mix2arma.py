"""
mix2arma.py - Convert Mixamo FBX animations to Arma 3 plain RTM (RTM_0101).

Retargets each Mixamo clip onto an Arma 3 OFP2_ManSkeleton armature that lives in
the .blend you open, then writes the RTM directly (no add-on needed for export).

Usage (Blender 2.93+ / 4.x / 5.x):
    Recommended - rig built from the Arma 3 Samples pivots (no .blend needed):
    blender -b --python mix2arma.py -- --pivots "<Arma 3 Samples>/Addons/TemplateRTM/ManSkeleton_Pivots.p3d" -i Walking.fbx -o Walking.rtm

    Or with your own rig in a .blend:
    blender -b ArmaRig.blend --python mix2arma.py -- -i Walking.fbx -o Walking.rtm
    blender -b ArmaRig.blend --python mix2arma.py -- -i mixamo_fbx/ -o rtm_out/ --root extract
    blender -b ArmaRig.blend --python mix2arma.py -- -i Rifle_Idle.fbx -o out/ --save-blend preview.blend

Target rig requirements:
    * An armature whose deform bones use OFP2_ManSkeleton names (Pelvis, Spine..Spine3, neck,
      neck1, head, LeftShoulder, LeftArm, LeftArmRoll, ..., RightToeBase). Case-insensitive.
    * Rest pose = Arma bind pose: bone HEADS must sit on the skeleton pivot points
      (ManSkeleton_Pivots.p3d from Arma 3 Samples). Bone roll/tail direction does not matter.
    * Blender axes = Arma axes with Y/Z swapped (Arma model space: origin at pelvis, feet ~-1 m,
      character faces -Z, i.e. -Y in Blender). --pivots builds exactly this.

Retarget method:
    World-space rotation transfer. For each mapped bone, the Mixamo bone's rotation delta from
    its rest pose is applied to the Arma bone, with a per-bone rest-alignment correction so a
    T-pose source drives an A-pose (or any other) target correctly. Bone lengths come from the
    Arma rig; only the pelvis takes translation (scaled by hip-height ratio). Unmapped Arma bones
    (roll bones, Spine3, neck1, weapon, launcher, camera, ...) follow their parent rigidly.

RTM matrix = lift @ pose_object_space @ rest_object_space^-1 (absolute model-space delta from rest;
lift raises the body so the feet stand on y=0, matching vanilla animations),
converted from Blender (right-handed, Z-up) to Arma (left-handed, Y-up, DirectX 4x3 row order).
"""

import argparse
import glob
import math
import os
import re
import struct
import sys

import bpy
from mathutils import Matrix, Vector

# --------------------------------------------------------------------------------------------
# Skeleton definition and bone map
# --------------------------------------------------------------------------------------------

OFP2_BONES = [
    "Pelvis", "Spine", "Spine1", "Spine2", "Spine3", "Camera", "weapon", "launcher",
    "neck", "neck1", "head",
    "Face_Hub", "Face_Jawbone", "Face_Jowl", "Face_chopRight", "Face_chopLeft",
    "Face_LipLowerMiddle", "Face_LipLowerLeft", "Face_LipLowerRight", "Face_Chin", "Face_Tongue",
    "Face_CornerRight", "Face_CheekSideRight", "Face_CornerLeft", "Face_CheekSideLeft",
    "Face_CheekFrontRight", "Face_CheekFrontLeft", "Face_CheekUpperRight", "Face_CheekUpperLeft",
    "Face_LipUpperMiddle", "Face_LipUpperRight", "Face_LipUpperLeft", "Face_NostrilRight",
    "Face_NostrilLeft", "Face_Forehead", "Face_BrowFrontRight", "Face_BrowFrontLeft",
    "Face_BrowMiddle", "Face_BrowSideRight", "Face_BrowSideLeft", "Face_Eyelids",
    "Face_EyelidUpperRight", "Face_EyelidUpperLeft", "Face_EyelidLowerRight",
    "Face_EyelidLowerLeft", "EyeLeft", "EyeRight",
]
for _s in ("Left", "Right"):
    OFP2_BONES += [f"{_s}Shoulder", f"{_s}Arm", f"{_s}ArmRoll", f"{_s}ForeArm",
                   f"{_s}ForeArmRoll", f"{_s}Hand", f"{_s}HandRing"]
    for _f in ("Ring", "Pinky", "Middle", "Index", "Thumb"):
        OFP2_BONES += [f"{_s}Hand{_f}{i}" for i in (1, 2, 3)]
    OFP2_BONES += [f"{_s}UpLeg", f"{_s}UpLegRoll", f"{_s}Leg", f"{_s}LegRoll",
                   f"{_s}Foot", f"{_s}ToeBase"]
OFP2_LOWER = {b.lower() for b in OFP2_BONES}
FACE_LOWER = {b.lower() for b in OFP2_BONES if b.startswith(("Face_", "Eye"))}


def build_bone_map():
    """(mixamo bone, arma bone, arma aim-child or None). Aim child defines bone direction
    for rest-pose alignment; None = inherit the alignment of the nearest mapped parent."""
    # Torso/head: no aim. Both rigs rest upright, and Arma's Pelvis and Spine pivots coincide,
    # so a direction-based correction there is meaningless (and can flip the pelvis).
    m = [
        ("Hips", "Pelvis", None),
        ("Spine", "Spine", None),
        ("Spine1", "Spine1", None),
        ("Spine2", "Spine2", None),
        ("Neck", "neck", None),
        ("Head", "head", None),
    ]
    for s in ("Left", "Right"):
        m += [
            (f"{s}Shoulder", f"{s}Shoulder", f"{s}Arm"),
            (f"{s}Arm", f"{s}Arm", f"{s}ForeArm"),
            (f"{s}ForeArm", f"{s}ForeArm", f"{s}Hand"),
            (f"{s}Hand", f"{s}Hand", f"{s}HandMiddle1"),
            (f"{s}UpLeg", f"{s}UpLeg", f"{s}Leg"),
            (f"{s}Leg", f"{s}Leg", f"{s}Foot"),
            (f"{s}Foot", f"{s}Foot", f"{s}ToeBase"),
            (f"{s}ToeBase", f"{s}ToeBase", None),
        ]
        for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for i in (1, 2, 3):
                aim = f"{s}Hand{f}{i + 1}" if i < 3 else None
                m.append((f"{s}Hand{f}{i}", f"{s}Hand{f}{i}", aim))
    return m


BONE_MAP = build_bone_map()

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def log(*a):
    print("[mix2arma]", *a, flush=True)


def strip_prefix(name):
    """'mixamorig:LeftArm' / 'mixamorig1:LeftArm' / 'Armature_mixamorig_LeftArm' -> 'LeftArm'."""
    name = name.split(":")[-1]
    return re.sub(r"^.*mixamorig\d*_", "", name)


def rot3(m):
    return m.to_3x3().normalized()


def to_rtm12(b):
    """Blender 4x4 (Z-up RH, column vectors) -> Arma 12 floats (Y-up LH, DirectX rows)."""
    return (b[0][0], b[2][0], b[1][0],
            b[0][2], b[2][2], b[1][2],
            b[0][1], b[2][1], b[1][1],
            b[0][3], b[2][3], b[1][3])


def name_field(s):
    raw = s.encode("ascii")[:31]
    return raw + b"\0" * (32 - len(raw))


def write_rtm(path, bone_names, frames, motion):
    """frames: list of (phase, [Matrix per bone in bone_names order])."""
    with open(path, "wb") as f:
        f.write(b"RTM_0101")
        f.write(struct.pack("<3f", *motion))
        f.write(struct.pack("<2I", len(frames), len(bone_names)))
        for n in bone_names:
            f.write(name_field(n))
        for phase, mats in frames:
            f.write(struct.pack("<f", phase))
            for n, m in zip(bone_names, mats):
                f.write(name_field(n))
                f.write(struct.pack("<12f", *to_rtm12(m)))
    expected = 8 + 12 + 8 + 32 * len(bone_names) + len(frames) * (4 + len(bone_names) * 80)
    actual = os.path.getsize(path)
    if actual != expected:
        raise RuntimeError(f"RTM size mismatch: {actual} != {expected}")


def forward_xy(left_head, right_head):
    left = left_head - right_head
    fwd = left.cross(Vector((0, 0, 1)))
    fwd.z = 0
    return fwd.normalized()


def signed_yaw(a, b):
    return math.atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y)


# --------------------------------------------------------------------------------------------
# Target rig
# --------------------------------------------------------------------------------------------


class TargetRig:
    def __init__(self, obj, include_face, all_bones):
        self.obj = obj
        self.mw_inv = obj.matrix_world.inverted()
        # Rig IK/Action/twist constraints are irrelevant to the solve (rest matrices only) but would
        # override baked keys in the --save-blend preview and can form dependency cycles. Mute them
        # in this session; the source .blend on disk is never overwritten.
        muted = 0
        for pb in obj.pose.bones:
            for c in pb.constraints:
                if not c.mute:
                    c.mute = True
                    muted += 1
        if muted:
            log(f"muted {muted} constraints on '{obj.name}' (control rig not used for export)")
        bones = obj.data.bones
        self.by_lower = {b.name.lower(): b.name for b in bones}
        missing = [n for n in ("pelvis", "leftupleg", "rightupleg") if n not in self.by_lower]
        if missing:
            raise RuntimeError(f"Target armature '{obj.name}' lacks Arma bones: {missing}")

        self.order = []  # parents before children (all rig bones)

        def walk(b):
            self.order.append(b.name)
            for c in b.children:
                walk(c)
        for b in bones:
            if b.parent is None:
                walk(b)

        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
        self.rest = {b.name: b.matrix_local.copy() for b in bones}
        self.rest_inv = {n: m.inverted() for n, m in self.rest.items()}
        self.head = {n: m.translation.copy() for n, m in self.rest.items()}

        def export_ok(n):
            low = n.lower()
            if not all_bones and low not in OFP2_LOWER:
                return False
            return include_face or low not in FACE_LOWER
        self.export = [n for n in self.order if export_ok(n)]

    def get(self, arma_name):
        return self.by_lower.get(arma_name.lower())


def find_target(rig_name):
    if rig_name:
        obj = bpy.data.objects.get(rig_name)
        if not obj or obj.type != "ARMATURE":
            raise RuntimeError(f"No armature named '{rig_name}'")
        return obj
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and any(b.name.lower() == "pelvis" for b in obj.data.bones):
            return obj
    raise RuntimeError("No Arma armature (with a 'Pelvis' bone) found in the open .blend; "
                       "use --rig, or --pivots to build one from the Arma 3 Samples")


def read_mlod_selections(path):
    """Named selections of the memory LOD of an unbinarized (MLOD) P3D: {name: [(x,y,z), ...]}."""
    d = open(path, "rb").read()
    if d[:4] != b"MLOD":
        raise RuntimeError(f"{path}: not an unbinarized MLOD p3d")

    def asciiz(o):
        e = d.index(b"\0", o)
        return d[o:e].decode("latin-1"), e + 1

    nlods, = struct.unpack_from("<I", d, 8)
    o, best = 12, None
    for _ in range(nlods):
        if d[o:o + 4] != b"P3DM":
            raise RuntimeError(f"{path}: unsupported LOD format")
        _a, _b, npts, nnrm, nfaces, _f = struct.unpack_from("<6I", d, o + 4)
        o += 28
        pts = [struct.unpack_from("<3f", d, o + 16 * i) for i in range(npts)]
        o += 16 * npts + 12 * nnrm
        for _ in range(nfaces):
            o += 72
            _, o = asciiz(o)
            _, o = asciiz(o)
        o += 4  # TAGG
        sels = {}
        while True:
            o += 1
            name, o = asciiz(o)
            size, = struct.unpack_from("<I", d, o)
            data = d[o + 4:o + 4 + size]
            o += 4 + size
            if name == "#EndOfFile#":
                break
            if not name.startswith("#"):
                sels[name] = [pts[i] for i in range(npts) if data[i]]
        res, = struct.unpack_from("<f", d, o)
        o += 4
        if res > 9e14:  # memory LOD
            best = sels
    if best is None:
        raise RuntimeError(f"{path}: no memory LOD")
    return best


def read_skeleton_cfg(path):
    txt = re.sub(r"//[^\n]*", "", open(path, encoding="latin-1").read())
    m = re.search(r"skeletonBones\[\]\s*=\s*\{(.*?)\}\s*;", txt, re.S)
    if not m:
        raise RuntimeError(f"{path}: no skeletonBones[] found")
    items = re.findall(r'"([^"]*)"', m.group(1))
    return list(zip(items[0::2], items[1::2]))


def build_rig_from_pivots(p3d_path, cfg_path):
    """Create an armature whose bone heads sit exactly on the Arma pivot points."""
    sels = {k.lower(): v for k, v in read_mlod_selections(p3d_path).items()}
    bones = read_skeleton_cfg(cfg_path)
    heads, no_pivot = {}, []
    for name, parent in bones:  # model.cfg lists parents before children
        pts = sels.get(name.lower())
        if pts:
            ax, ay, az = (sum(p[i] for p in pts) / len(pts) for i in range(3))
            heads[name] = Vector((ax, az, ay))  # Arma (x, y-up, z) -> Blender (x, y, z-up)
        else:
            heads[name] = heads[parent].copy() if parent else Vector()
            no_pivot.append(name)
    arm = bpy.data.armatures.new("ArmaPivotsRig")
    obj = bpy.data.objects.new("ArmaPivotsRig", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for name, parent in bones:
        b = arm.edit_bones.new(name)
        b.head = heads[name]
        b.tail = heads[name] + Vector((0, 0, 0.05))
        if parent:
            b.parent = eb[parent]
        eb[name] = b
    bpy.ops.object.mode_set(mode="OBJECT")
    shown = [n for n in no_pivot if not n.lower().startswith(("face_", "eye"))]
    log(f"built rig from pivots: {len(bones)} bones; no pivot (follow parent): {', '.join(shown)}")
    return obj


# --------------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------------


def import_fbx(path):
    before = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=path, use_anim=True, automatic_bone_orientation=False,
                             ignore_leaf_bones=True)
    new_objs = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new_objs if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"No armature in {path}")
    src = max(arms, key=lambda o: len(o.data.bones))
    new_actions = [a for a in bpy.data.actions if a not in before_actions]
    if new_actions and (src.animation_data is None or src.animation_data.action is None):
        if src.animation_data is None:
            src.animation_data_create()
        src.animation_data.action = max(new_actions, key=lambda a: len(a.fcurves)
                                        if hasattr(a, "fcurves") else 0)
    return src, new_objs, new_actions


def cleanup(objs, actions):
    datas = []
    for o in objs:
        if o.data is not None:
            datas.append(o.data)
        bpy.data.objects.remove(o, do_unlink=True)
    for d in datas:
        try:
            if d.users == 0:
                if isinstance(d, bpy.types.Armature):
                    bpy.data.armatures.remove(d)
                elif isinstance(d, bpy.types.Mesh):
                    bpy.data.meshes.remove(d)
        except ReferenceError:
            pass
    for a in actions:
        try:
            bpy.data.actions.remove(a)
        except ReferenceError:
            pass


def convert(src, tgt, args):
    scene = bpy.context.scene
    src_bones = {strip_prefix(b.name): b.name for b in src.data.bones}

    # Resolve map against what actually exists on both rigs.
    pairs = {}   # arma bone -> mixamo bone (real names)
    aims = {}    # arma bone -> arma aim bone (real name)
    for mx, ar, aim in BONE_MAP:
        t, s = tgt.get(ar), src_bones.get(mx)
        if t and s:
            pairs[t] = s
            if aim and tgt.get(aim):
                aims[t] = tgt.get(aim)
    rev = {v: k for k, v in pairs.items()}
    root_t = tgt.get("Pelvis")
    if root_t not in pairs:
        raise RuntimeError("Source has no Hips bone (is this a Mixamo skeleton?)")
    unmapped_src = sorted(set(src_bones) - {strip_prefix(s) for s in pairs.values()})
    log(f"mapped {len(pairs)} bones; source bones not used: {', '.join(unmapped_src) or 'none'}")

    # Source -> target-object-space transform, including facing correction.
    s0 = tgt.mw_inv @ src.matrix_world
    srest = {n: b.matrix_local.copy() for n, b in src.data.bones.items()}

    def shead(n, S):
        return (S @ srest[n]).translation

    sl, sr = src_bones.get("LeftUpLeg"), src_bones.get("RightUpLeg")
    tl, tr = tgt.get("LeftUpLeg"), tgt.get("RightUpLeg")
    yaw = 0.0
    if sl and sr:
        yaw = signed_yaw(forward_xy(shead(sl, s0), shead(sr, s0)),
                         forward_xy(tgt.head[tl], tgt.head[tr]))
    yaw = math.radians(90) * round(yaw / math.radians(90))  # rigs face along an axis; drop hip asymmetry
    S = Matrix.Rotation(yaw, 4, "Z") @ s0
    log(f"facing correction: {math.degrees(yaw):.1f} deg")

    hips = pairs[root_t]
    src_hip_h = shead(hips, S).z
    # Hip height above the ground for both rigs (Arma's origin is at the pelvis, not the feet).
    feet = ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")
    tg = [tgt.head[tgt.get(n)].z for n in feet if tgt.get(n)]
    sg = [shead(src_bones[n], S).z for n in feet if n in src_bones]
    t_h = tgt.head[root_t].z - (min(tg) if tg else 0.0)
    s_h = src_hip_h - (min(sg) if sg else 0.0)
    scale = args.scale or (t_h / s_h if abs(s_h) > 1e-6 else 1.0)
    log(f"root translation scale: {scale:.4f}")

    src_rest_rot_inv = {s: rot3(S @ srest[s]).inverted() for s in pairs.values()}

    # Rest-alignment corrections C (target rest dir -> source rest dir).
    C = {}
    for t in tgt.order:
        if t not in pairs:
            continue
        c = None
        if args.align_rest and t in aims and aims[t] in pairs:
            a = aims[t]
            tdir = tgt.head[a] - tgt.head[t]
            sdir = shead(pairs[a], S) - shead(pairs[t], S)
            if tdir.length > 1e-6 and sdir.length > 1e-6:
                q = tdir.rotation_difference(sdir)
                if q.angle > math.radians(100):
                    log(f"WARNING: rest pose of '{t}' differs by {math.degrees(q.angle):.0f} deg "
                        f"from source; ignoring (check the rig's bone positions)")
                else:
                    c = q.to_matrix()
        if c is None:  # inherit nearest mapped ancestor's correction
            p = tgt.parent[t]
            while p and p not in C:
                p = tgt.parent[p]
            c = C[p].copy() if p else Matrix.Identity(3)
        C[t] = c

    # Frame range.
    action = src.animation_data.action if src.animation_data else None
    if action is None:
        raise RuntimeError("Imported armature has no animation")
    f0, f1 = (int(round(x)) for x in action.frame_range)
    if args.frame_start is not None:
        f0 = args.frame_start
    if args.frame_end is not None:
        f1 = args.frame_end
    frames = list(range(f0, f1 + 1, args.step))
    if frames[-1] != f1:
        frames.append(f1)
    if len(frames) < 2:
        raise RuntimeError("Animation needs at least 2 frames")

    # Pass 1: sample source.
    samples = []
    hip_rest = shead(hips, S)
    for f in frames:
        scene.frame_set(f)
        rots = {s: rot3(S @ src.pose.bones[s].matrix) for s in pairs.values()}
        off = ((S @ src.pose.bones[hips].matrix).translation - hip_rest) * scale
        samples.append((rots, off))

    # Root motion handling.
    n = len(frames)
    phases = [(f - frames[0]) / (frames[-1] - frames[0]) for f in frames]
    motion_b = Vector((0, 0, 0))
    if args.root == "extract":
        total = samples[-1][1] - samples[0][1]
        total.z = 0
        start = samples[0][1].copy()
        start.z = 0
        for i in range(n):
            samples[i] = (samples[i][0], samples[i][1] - start - total * phases[i])
        motion_b = total
    motion_arma = (motion_b.x, motion_b.z, motion_b.y)

    # Arma places the model origin (pelvis) on the ground, so animations must lift the body until
    # the feet touch y=0 - vanilla standing idles carry ~+1.0 m on every bone.
    lift = args.ground_lift if args.ground_lift is not None else (-min(tg) if tg else 0.0)
    L = Matrix.Translation((0, 0, lift))
    log(f"ground lift: {lift:.3f} m")

    # Pass 2: solve target pose per frame.
    out_frames, pose_mats = [], []
    for i in range(n):
        rots, off = samples[i]
        M = {}
        for t in tgt.order:
            rest = tgt.rest[t]
            p = tgt.parent[t]
            if t in pairs:
                s = pairs[t]
                D = rots[s] @ src_rest_rot_inv[s]
                m = (D @ C[t] @ rot3(rest)).to_4x4()
                if t == root_t or p is None:
                    m.translation = tgt.head[t] + (off if t == root_t else Vector())
                else:
                    m.translation = (M[p] @ tgt.rest_inv[p] @ rest).translation
            else:
                m = (M[p] @ tgt.rest_inv[p] @ rest) if p else rest.copy()
            M[t] = m
        out_frames.append((phases[i], [L @ M[t] @ tgt.rest_inv[t] for t in tgt.export]))
        pose_mats.append(M)

    return frames, out_frames, pose_mats, motion_arma


def bake_preview(tgt, frames, pose_mats, name):
    """Keyframe the solved pose onto the target rig as an action (for visual checking)."""
    obj = tgt.obj
    if obj.animation_data is None:
        obj.animation_data_create()
    act = bpy.data.actions.new(name)
    obj.animation_data.action = act
    for f, M in zip(frames, pose_mats):
        for t in tgt.order:
            p = tgt.parent[t]
            if p:
                basis = (tgt.rest_inv[p] @ tgt.rest[t]).inverted() @ (M[p].inverted() @ M[t])
            else:
                basis = tgt.rest_inv[t] @ M[t]
            pb = obj.pose.bones[t]
            pb.rotation_mode = "QUATERNION"
            pb.matrix_basis = basis
            pb.keyframe_insert("location", frame=f, group=t)
            pb.keyframe_insert("rotation_quaternion", frame=f, group=t)
    act.use_fake_user = True


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="mix2arma", description=__doc__.split("\n")[1])
    ap.add_argument("-i", "--input", required=True, help="Mixamo .fbx file or folder of .fbx")
    ap.add_argument("-o", "--output", required=True, help=".rtm file, or folder for batch")
    ap.add_argument("--rig", help="Target armature object name (default: first with a Pelvis bone)")
    ap.add_argument("--pivots", help="ManSkeleton_Pivots.p3d from Arma 3 Samples: build the target rig "
                                     "from it instead of using a rig in the .blend (recommended)")
    ap.add_argument("--model-cfg", help="model.cfg with the skeleton (default: next to --pivots)")
    ap.add_argument("--root", choices=("keep", "extract"), default="keep",
                    help="keep: root motion stays in Pelvis. extract: horizontal travel is removed "
                         "from Pelvis and written as the RTM motion vector (use for walk/run cycles "
                         "NOT downloaded 'In Place')")
    ap.add_argument("--scale", type=float, help="Root translation scale (default: hip-height ratio)")
    ap.add_argument("--ground-lift", type=float,
                    help="Vertical lift in metres so feet sit on the ground (default: auto from toe "
                         "pivots, ~1.0 for the Arma man skeleton)")
    ap.add_argument("--fps", type=int, default=30, help="Scene fps for FBX import (default 30)")
    ap.add_argument("--step", type=int, default=1, help="Sample every Nth frame (default 1)")
    ap.add_argument("--frame-start", type=int)
    ap.add_argument("--frame-end", type=int)
    ap.add_argument("--no-align-rest", dest="align_rest", action="store_false",
                    help="Disable rest-pose alignment (only if both rigs share the same rest pose)")
    ap.add_argument("--include-face", action="store_true", help="Also write Face_*/Eye bones")
    ap.add_argument("--all-bones", action="store_true",
                    help="Write every rig bone, not just OFP2_ManSkeleton bones")
    ap.add_argument("--lowercase", action="store_true", help="Write bone names in lowercase")
    ap.add_argument("--save-blend", help="Bake results onto the Arma rig and save this .blend")
    return ap.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    if os.path.isdir(args.input):
        inputs = sorted(glob.glob(os.path.join(args.input, "*.fbx")))
        out_dir = args.output
    else:
        inputs = [args.input]
        out_dir = None if args.output.lower().endswith(".rtm") else args.output
    if not inputs:
        raise SystemExit(f"No .fbx files in {args.input}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpy.context.scene.render.fps = args.fps
    bpy.context.scene.render.fps_base = 1.0
    if args.pivots:
        cfg = args.model_cfg or os.path.join(os.path.dirname(os.path.abspath(args.pivots)), "model.cfg")
        rig_obj = build_rig_from_pivots(args.pivots, cfg)
    else:
        rig_obj = find_target(args.rig)
    tgt = TargetRig(rig_obj, args.include_face, args.all_bones)
    log(f"target rig '{tgt.obj.name}': exporting {len(tgt.export)} bones")

    failures = 0
    for path in inputs:
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, base + ".rtm") if out_dir else args.output
        log(f"--- {path}")
        objs, actions = [], []
        try:
            src, objs, actions = import_fbx(path)
            frames, out_frames, pose_mats, motion = convert(src, tgt, args)
            names = [n.lower() if args.lowercase else n for n in tgt.export]
            write_rtm(out, names, out_frames, motion)
            dur = (frames[-1] - frames[0]) / args.fps
            log(f"wrote {out}: {len(out_frames)} frames, {len(names)} bones, "
                f"motion=({motion[0]:.3f}, {motion[1]:.3f}, {motion[2]:.3f}), duration {dur:.3f}s")
            log(f'config hint: file = "\\yourAddon\\anims\\{base}.rtm"; speed = -{dur:.3f};')
            if args.save_blend:
                bake_preview(tgt, frames, pose_mats, base)
        except Exception as e:  # keep batch going
            failures += 1
            log(f"FAILED {path}: {e}")
        finally:
            cleanup(objs, actions)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend), copy=True)
        log(f"saved preview {args.save_blend}")
    log(f"done: {len(inputs) - failures}/{len(inputs)} converted")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
