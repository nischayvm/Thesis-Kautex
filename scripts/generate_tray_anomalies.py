"""Generate seeded appearance defects on the inspection-cell tray.
Run this text in Blender to regenerate from the Anomaly controls properties.
For a batch, call generate_anomalies(seed=..., counts=...) before each capture.
Anomaly sizes are synthetic visual ranges, in tray-local metres.
"""
import bpy
import json
import math
import random
from itertools import groupby
from mathutils import Vector

CONTROL = "Anomaly controls"
MANIFEST = "anomaly_instances.json"
KERNEL = "Anomaly · irregular local footprint"
MATERIALS = ("PA6-GF30 · Black matte mould texture",
             "Cross-ply organosheet · glossy resin, twill and frost")
AOVS = {"blister": "Defect_Blister", "burn": "Defect_Burn", "sink": "Defect_Sink"}
COUNT_KEYS = {"flange_blisters": "Flange blisters", "crossply_blisters": "Cross-ply blisters",
              "burns": "Curved-edge burns", "flange_sinks": "Flange sinks", "crossply_sinks": "Cross-ply sinks"}
DEFAULT_COUNTS = dict(zip(COUNT_KEYS, (5, 6, 3, 2, 3)))
GAINS = {"blister": "Blister amount", "burn": "Burn amount", "sink": "Sink amount"}
MIN_Z = {"flange": .104, "cross-ply": .0045}
FLANGE_X = (-.663, -.620, -.390, -.252, -.124, .124, .252, .390, .620, .663)
MODULE_X = (-.619, -.378, -.126, .126, .378, .619)


def anomaly_control(scene):
    ctrl = bpy.data.objects.get(CONTROL)
    if ctrl is not None:
        return ctrl
    ctrl = bpy.data.objects.new(CONTROL, None)
    bpy.data.collections["Lighting controls"].objects.link(ctrl)
    ctrl.empty_display_type = 'PLAIN_AXES'
    ctrl.empty_display_size = .055
    ctrl.location = (1.22, .335, 1.15)
    ctrl["Seed"] = 24117
    ctrl.id_properties_ui("Seed").update(min=0, max=2147483647,
                                           description="Run generate_tray_anomalies.py again after changing the seed.")
    for k, label in COUNT_KEYS.items():
        ctrl[label] = DEFAULT_COUNTS[k]
        ctrl.id_properties_ui(label).update(min=0, max=24,
                                             description="Number to place when the generator runs again.")
    for label in ("Enabled", *GAINS.values()):
        ctrl[label] = 1.0
        ctrl.id_properties_ui(label).update(min=0.0, max=1.0,
                                             description="Live material control. Zero suppresses this response.")
    return ctrl


def crossply_blister_sites(tray, seed, count, other_defects):
    """Fit loose laminate lifts within bays and across selected shallow ribs."""
    if not count:
        return []
    rng = random.Random(seed ^ 0xBCC017)
    host = tray.evaluated_get(bpy.context.evaluated_depsgraph_get())
    z = MIN_Z["cross-ply"]

    def top_at(x, y):
        hit, point, normal, face = host.ray_cast((x, y, z + .13), (0, 0, -1))
        if hit:
            return point.z, normal.z, host.data.polygons[face].material_index
        return -1., 0., -1

    def floor_at(x, y):
        height, normal_z, material = top_at(x, y)
        return abs(height-z) < .00015 and normal_z > .998 and material == 1

    def spans(start, end, probe):
        samples = (i * .001 for i in range(start, end))
        for valid, group in groupby(samples, probe):
            points = list(group)
            if valid and points[-1] - points[0] >= .054:
                yield points[0], points[-1]

    xs = list(spans(-740, 741, lambda x: floor_at(x, .035)))
    ys = list(spans(-335, 336, lambda y: floor_at(-.575, y)))
    bays = [((x0, x1, y0, y1), None) for x0, x1 in xs for y0, y1 in ys]
    crossings = []
    for axis, (spans_on_axis, across_spans) in enumerate(((xs,ys),(ys,xs))):
        for (start, left), (right, end) in zip(spans_on_axis, spans_on_axis[1:]):
            if right-left > .009:
                continue
            middle = (left+right)*.5
            rib_top = top_at(middle,.035)[0] if axis == 0 else top_at(-.575,middle)[0]
            if not z+.004 <= rib_top <= z+.022:
                continue
            for low, high in across_spans:
                bounds = (start,end,low,high) if axis == 0 else (low,high,start,end)
                crossings.append((bounds,(axis,left,right,rib_top)))
    rng.shuffle(crossings)
    rng.shuffle(bays)
    sinks = [d for d in other_defects if d["surface"] == "cross-ply" and d["class"] == "sink"]
    used_ids = {d["id"] for d in other_defects}
    index = 1
    placed = []
    max_crossings = min(10, math.ceil(count*.6))
    cross_count = 0
    clearance = []
    for (x0, x1, y0, y1), rib in crossings+bays:
        if rib and cross_count >= max_crossings:
            continue
        w, h = x1-x0, y1-y0
        x = (x0+x1)*.5 + rng.uniform(-.035,.035)*w
        y = (y0+y1)*.5 + rng.uniform(-.035,.035)*h
        if rib:
            axis,left,right,rib_top = rib
            offset = (left+right)*.5 + rng.choice((-1,1))*rng.uniform(.006,.024)
            if axis == 0:
                x = offset
            else:
                y = offset

        def supported(px, py):
            if rib and left-.0005 <= (px,py)[axis] <= right+.0005:
                height, _, _ = top_at(px,py)
                return z-.00015 <= height <= rib_top+.0015
            return floor_at(px,py)

        max_x, max_y = min(x-x0,x1-x)-.003, min(y-y0,y1-y)-.003
        if rib and axis == 0:
            max_x = min(max_x,w*rng.uniform(.28,.42))
        if rib and axis == 1:
            max_y = min(max_y,h*rng.uniform(.28,.42))
        for shrink in (1.0, .92, .84):
            dx, dy = max_x*shrink, max_y*shrink
            if rib and (dx,dy)[axis] - abs((x,y)[axis]-(left+right)*.5) < .020:
                continue
            if any(abs(x-px) < dx+sx+.003 and abs(y-py) < dy+sy+.003
                   for px,py,sx,sy in clearance):
                continue
            if any(abs(x-d["centre_local_m"][0]) < dx+1.15*max(d["radii_m"])
                   and abs(y-d["centre_local_m"][1]) < dy+1.15*max(d["radii_m"]) for d in sinks):
                continue
            nx, ny = math.ceil(2*dx/.003), math.ceil(2*dy/.003)
            probes_x = [x+dx*(2*i/nx-1) for i in range(nx+1)]
            probes_y = [y+dy*(2*i/ny-1) for i in range(ny+1)]
            if rib:
                (probes_x,probes_y)[axis].append((left+right)*.5)
            if all(supported(px,py) for px in probes_x for py in probes_y):
                break
        else:
            continue
        fraction = rng.uniform(.85,.97)
        rx, ry = dx*fraction, dy*fraction*rng.uniform(.94,1.0)
        while f"blister-{index:03d}" in used_ids:
            index += 1
        defect = {
            "id": f"blister-{index:03d}", "class": "blister", "surface": "cross-ply",
            "centre_local_m": [x,y,z], "radii_m": [rx,ry], "angle_deg": 0.0,
            "relief_m": rng.uniform(.0027,.0054), "near_hole_local_m": None, "corner": None,
            "field_seed": rng.uniform(0,2000), "irregularity": rng.uniform(.09,.20),
            "bay_bounds_local_m": [x0,x1,y0,y1],
            "maximum_width_factor": min(dx/rx,dy/ry),
        }
        if rib:
            defect["crossed_rib"] = {"axis": "XY"[axis], "gap_local_m": [left,right],
                                     "top_local_m": rib_top}
            cross_count += 1
        placed.append(defect)
        clearance.append((x,y,dx,dy))
        index += 1
        if len(placed) == count:
            return placed
    raise ValueError(f"Found only {len(placed)} clear locations for {count} cross-ply blisters.")


def sites(tray, seed, counts):
    rng = random.Random(seed)
    evaluated = tray.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.data
    placed = []

    def sample(kind, region, x, y, major, minor, angle, height, near=None, corner=None):
        z = MIN_Z[region]
        for other in placed:
            if other["surface"] == region and math.hypot(x-other["centre_local_m"][0],
                    y-other["centre_local_m"][1]) < .79*(max(major,minor)+max(other["radii_m"])):
                return False

        # Reject sites that clip a bore, rib, boss or boundary. Deliberate scorch marks
        # can reach the curved outer flange boundary, so only the anchor must sit on top.
        probes = [0] if kind == 'burn' else [0, *range(1, 13)]
        for i in probes:
            a = math.tau*i/12
            u = 1.14*major*math.cos(a) if i else 0
            v = 1.14*minor*math.sin(a) if i else 0
            local_x = x+u*math.cos(angle)-v*math.sin(angle)
            local_y = y+u*math.sin(angle)+v*math.cos(angle)
            hit, p, n, face = evaluated.ray_cast(Vector((local_x,local_y,z+.010)),
                                           Vector((0,0,-1)), distance=.013)
            if not hit or face < 0 or abs(p.z-z)>.00015 or n.z<.998:
                return False
            roof_material = mesh.polygons[face].material_index
            if roof_material != int(region == "cross-ply"):
                return False
        index = len(placed)+1
        placed.append({
            "id": f"{kind}-{index:03d}", "class": kind, "surface": region,
            "centre_local_m": [x,y,z], "radii_m": [major,minor],
            "angle_deg": math.degrees(angle), "relief_m": height,
            "near_hole_local_m": near, "corner": corner, "field_seed": rng.uniform(0,2000),
            "irregularity": rng.uniform(.11,.34) if kind!="sink" else rng.uniform(.065,.15)
        })
        return True

    def choose(n, picker):
        for i in range(n):
            for _ in range(1600):
                if sample(*picker(i,n)):
                    break
            else:
                raise ValueError(f"Could not place requested defect {i+1} of {n} without clipping the tray features.")

    def flange_blister(i,n):
        major,minor=rng.uniform(.0028,.0085),rng.uniform(.0015,.0044)
        angle=rng.uniform(-.66,.66)
        side=rng.choice((-1,1))
        pin=None
        if i < min(3,n):
            hx=rng.choice(FLANGE_X)
            hy=side*.324
            x=hx+rng.choice((-1,1))*(.0065+1.18*major+rng.uniform(.0015,.0033))
            y=hy+side*rng.uniform(-.0036,.0045)
            pin=[hx,hy,.104]
        else:
            x=rng.uniform(-.635,.635)
            y=side*rng.uniform(.316,.334)
        rise=rng.uniform(.00023,.00103)
        return ("blister","flange",x,y,major,minor,angle,rise,pin)

    corners=[(-1,-1),(1,-1),(-1,1),(1,1)]
    rng.shuffle(corners)
    def burn(i,n):
        sx,sy=corners[i%len(corners)]
        theta=rng.uniform(.32,1.25)
        rad=rng.uniform(.044,.050)
        x=sx*(.698+rad*math.cos(theta))
        y=sy*(.290+rad*math.sin(theta))
        tangent=math.atan2(sy*math.cos(theta),-sx*math.sin(theta))
        return ("burn","flange",x,y,rng.uniform(.013,.030),rng.uniform(.004,.009),
                tangent,-rng.uniform(.000010,.000045),None,[sx,sy])

    def flange_sink(i,n):
        x=rng.choice((-.563,-.455,-.396,-.318,-.174,-.088,.088,.174,.318,.396,.455,.563)) + rng.uniform(-.004,.004)
        y=rng.choice((-1,1))*rng.uniform(.315,.326)
        return ("sink","flange",x,y,rng.uniform(.007,.017),rng.uniform(.0034,.0061),
                rng.uniform(-.40,.40),-rng.uniform(.00016,.00063))

    def crossply_sink(i,n):
        bx=rng.choice(MODULE_X)
        by=rng.choice((-1,1))*.139
        x=bx+rng.choice((-1,1))*rng.uniform(.015,.031)
        y=by+rng.choice((-1,1))*rng.uniform(.013,.034)
        return ("sink","cross-ply",x,y,rng.uniform(.008,.018),rng.uniform(.0038,.009),
                rng.uniform(-.70,.70),-rng.uniform(.00024,.00086))

    for k, fn in (("flange_blisters",flange_blister),("burns",burn),
                  ("flange_sinks",flange_sink),("crossply_sinks",crossply_sink)):
        choose(counts[k],fn)
    placed.extend(crossply_blister_sites(tray, seed, counts["crossply_blisters"], placed))
    return placed


def node_ops(tree):
    def node(kind,label,x,y):
        ob=tree.nodes.new(kind)
        ob.label=label
        ob.location=(x,y)
        return ob
    def put(socket,val):
        if isinstance(val,bpy.types.NodeSocket):
            tree.links.new(val,socket)
        else:
            socket.default_value=val
    def op(kind,label,a,b,x,y):
        ob=node('ShaderNodeMath',label,x,y)
        ob.operation=kind
        put(ob.inputs[0],a)
        if b is not None:
            put(ob.inputs[1],b)
        return ob.outputs[0]
    def remap(label,value,source_low,source_high,target_low,target_high,x,y,smooth=True):
        ob=node('ShaderNodeMapRange',label,x,y)
        ob.clamp=True
        ob.interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR'
        for k,v in zip(("Value","From Min","From Max","To Min","To Max"),
                       (value,source_low,source_high,target_low,target_high)):
            put(ob.inputs[k],v)
        return ob.outputs[0]
    return node,put,op,remap


def field_kernel():
    prior=bpy.data.node_groups.get(KERNEL)
    if prior:
        return prior
    g=bpy.data.node_groups.new(KERNEL,'ShaderNodeTree')
    for name,typ,direction in (
        ("Point","NodeSocketVector",'INPUT'),("Centre","NodeSocketVector",'INPUT'),
        ("Tangent","NodeSocketVector",'INPUT'),("Across","NodeSocketVector",'INPUT'),
        ("Radius U","NodeSocketFloat",'INPUT'),("Radius V","NodeSocketFloat",'INPUT'),
        ("Depth range","NodeSocketFloat",'INPUT'),("Variation","NodeSocketFloat",'INPUT'),
        ("Seed","NodeSocketFloat",'INPUT'),("Footprint","NodeSocketFloat",'OUTPUT'),
        ("Profile","NodeSocketFloat",'OUTPUT')):
        g.interface.new_socket(name=name,in_out=direction,socket_type=typ)
    node,put,op,remap=node_ops(g)
    ins=node('NodeGroupInput',"Instance controls",-1220,200).outputs
    position=node('ShaderNodeVectorMath',"Tray-local position relative to the centre",-960,160)
    position.operation='SUBTRACT'
    put(position.inputs[0],ins["Point"])
    put(position.inputs[1],ins["Centre"])
    axes=[]
    for i,(axis,rad) in enumerate((("Tangent","Radius U"),("Across","Radius V"))):
        dot=node('ShaderNodeVectorMath',axis,-730,350-i*220)
        dot.operation='DOT_PRODUCT'
        put(dot.inputs[0],position.outputs["Vector"])
        put(dot.inputs[1],ins[axis])
        axes.append(op('DIVIDE',rad,dot.outputs["Value"],ins[rad],-500,350-i*220))
    plane=node('ShaderNodeCombineXYZ',"Normalized tangent coordinates",-270,270)
    put(plane.inputs["X"],axes[0])
    put(plane.inputs["Y"],axes[1])
    length=node('ShaderNodeVectorMath',"Ellipse radius",-50,430)
    length.operation='LENGTH'
    put(length.inputs[0],plane.outputs[0])
    noise=node('ShaderNodeTexNoise',"Asymmetric contour", -50,120)
    noise.noise_dimensions='4D'
    noise.inputs["Scale"].default_value=1.85
    noise.inputs["Detail"].default_value=2
    noise.inputs["Roughness"].default_value=.59
    put(noise.inputs["Vector"],plane.outputs[0])
    put(noise.inputs["W"],ins["Seed"])
    variation=op('SUBTRACT',"Centred contour variation",noise.outputs["Fac"],.5,190,130)
    variation=op('MULTIPLY',"Contour irregularity",variation,ins["Variation"],380,130)
    border=op('ADD',"Varied radius",1.0,variation,570,130)
    distance=op('DIVIDE',"Irregular distance",length.outputs["Value"],border,780,430)
    profile=remap("Rounded relief, zero at the edge",distance,0.,1.,1.,0.,980,425)
    mask=remap("Soft irregular footprint",distance,.72,1.0,1.,0.,980,150)
    separation=node('ShaderNodeSeparateXYZ',"Vertical locality",-730,-200)
    put(separation.inputs[0],position.outputs["Vector"])
    absolute=op('ABSOLUTE',"Distance from the host surface",separation.outputs["Z"],None,-500,-200)
    z=remap("Constrain to one surface",absolute,0.,ins["Depth range"],1.,0.,-270,-200)
    profile=op('MULTIPLY',"Localized profile",profile,z,1190,425)
    mask=op('MULTIPLY',"Localized footprint",mask,z,1190,150)
    out=node('NodeGroupOutput',"Local anomaly result",1420,350)
    put(out.inputs["Profile"],profile)
    put(out.inputs["Footprint"],mask)
    return g


def shader_controls(tree, ctrl, node, put, op, y):
    def control(key,row):
        item=node('ShaderNodeValue',key,-1340,row)
        item.outputs[0].default_value=float(ctrl[key])
        curve=item.outputs[0].driver_add("default_value")
        drv=curve.driver
        var=drv.variables.new()
        var.name="value"
        var.type='SINGLE_PROP'
        var.targets[0].id=ctrl
        var.targets[0].data_path=f'["{key}"]'
        drv.expression="value"
        return item.outputs[0]
    enabled=control("Enabled",y)
    return {k:op('MULTIPLY',"Enable "+k,enabled,control(label,y-(j+1)*125),-1110,y-(j+1)*125)
            for j,(k,label) in enumerate(GAINS.items())}


def anomaly_material(clean, region, defects, ctrl):
    material=clean.copy()
    material.name="Tray anomalies · "+region
    tree=material.node_tree
    base_shaders=[n for n in tree.nodes if n.type=='BSDF_PRINCIPLED']
    node,put,op,remap=node_ops(tree)
    y=min(n.location.y for n in tree.nodes)-420
    position=node('ShaderNodeTexCoord',"Tray-local defect coordinates",-1340,y+220).outputs["Object"]
    ctrl_signal=shader_controls(tree,ctrl,node,put,op,y)
    kernel=field_kernel()
    masks=dict.fromkeys(AOVS,0.0)
    profiles=dict.fromkeys(AOVS,0.0)
    heights=dict.fromkeys(AOVS,0.0)
    for i,defect in enumerate(d for d in defects if d["surface"]==region):
        k=defect["class"]
        if k=="blister":
            continue  # The raised mesh carries the blister relief and mask.
        x=-860+560*(i%4)
        row=y-580-500*(i//4)
        block=node('ShaderNodeGroup',defect["id"],x,row)
        block.node_tree=kernel
        angle=math.radians(defect["angle_deg"])
        for name,value in (
            ("Point",position),("Centre",defect["centre_local_m"]),
            ("Tangent",(math.cos(angle),math.sin(angle),0)),
            ("Across",(-math.sin(angle),math.cos(angle),0)),
            ("Radius U",defect["radii_m"][0]),("Radius V",defect["radii_m"][1]),
            ("Depth range",.006 if k=='burn' else .0017),
            ("Variation",defect["irregularity"]),("Seed",defect["field_seed"])):
            put(block.inputs[name],value)
        masks[k]=op('MAXIMUM',k+" class mask",masks[k],block.outputs["Footprint"],x+205,row)
        profiles[k]=op('MAXIMUM',k+" class shape",profiles[k],block.outputs["Profile"],x+380,row)
        h=op('MULTIPLY',"Relief in metres",block.outputs["Profile"],defect["relief_m"],x+205,row-155)
        heights[k]=op('ADD',"Accumulate signed relief",heights[k],h,x+380,row-155)

    def masked(kind,source,label,x,row):
        return op('MULTIPLY',label,source[kind],ctrl_signal[kind],x,row)
    for j,(kind,name) in enumerate(AOVS.items()):
        aov=node('ShaderNodeOutputAOV',name,1600,y-j*155)
        aov.aov_name=name
        put(aov.inputs["Value"],masked(kind,masks,"Class mask",1390,y-j*155))
    reliefs=[masked(k,heights,"Class relief",1860,y-j*155) for j,k in enumerate(('burn','sink'))]
    merged=op('ADD',"Burn and sink relief",*reliefs,2260,y)
    burn=masked('burn',profiles,"Char core",1870,y-710)
    burn_edge=masked('burn',masks,"Irregular burn perimeter",1870,y-870)
    sink=masked('sink',profiles,"Sink centre",1870,y-1030)

    def initial(socket):
        if socket.is_linked:
            return socket.links[0].from_socket
        current=socket.default_value
        return tuple(current) if hasattr(current,'__len__') else current

    def mix_rgb(label,factor,a,b,x,row):
        mx=node('ShaderNodeMixRGB',label,x,row)
        put(mx.inputs[0],factor)
        put(mx.inputs[1],a)
        put(mx.inputs[2],b)
        return mx.outputs[0]

    for j,bsdf in enumerate(base_shaders):
        row=y-450-(j*1060)
        normal=bsdf.inputs["Normal"]
        input_normal=initial(normal) if normal.is_linked else None
        bump=node('ShaderNodeBump',"Continuous relief after base texture",2760,row)
        bump.inputs["Strength"].default_value=1.0
        bump.inputs["Distance"].default_value=1.0
        put(bump.inputs["Height"],merged)
        if input_normal is not None:
            put(bump.inputs["Normal"],input_normal)
        put(normal,bump.outputs["Normal"])
        if bsdf.inputs["Coat Weight"].default_value>0:
            put(bsdf.inputs["Coat Normal"],bump.outputs["Normal"])
        if region=="flange":
            base_color=initial(bsdf.inputs["Base Color"])
            warm=op('MULTIPLY',"Soot halo weight",burn_edge,.78,2310,row-495)
            after=mix_rgb("Warm char halo",warm,base_color,(.035,.012,.004,1),2750,row-290)
            black=op('MULTIPLY',"Dark core weight",burn,.97,2530,row-510)
            after=mix_rgb("Dark char core",black,after,(.0018,.00105,.00065,1),2970,row-290)
            put(bsdf.inputs["Base Color"],after)
        old=initial(bsdf.inputs["Roughness"])
        additions=op('MULTIPLY',"Rougher char",burn_edge,.32,2310,row-910)
        rough=op('ADD',"Roughened burn",old,additions,2760,row-710)
        gloss=op('MULTIPLY',"Soft sink highlight",sink,-.065,2530,row-910)
        rough=op('ADD',"Final defect roughness",rough,gloss,2980,row-710)
        put(bsdf.inputs["Roughness"],rough)
    return material



BLISTER_COLLECTION = "Raised cross-ply laminate"
BLISTER_MATERIAL = "Cross-ply · raised laminate"
FLANGE_BLISTER_COLLECTION = "Raised flange blisters"
FLANGE_BLISTER_MATERIAL = "PA6-GF30 · raised flange skin"
FOOTPRINT_ATTRIBUTE = "Laminate blister footprint"


def configure_blister_scale(ctrl, objects, defects, scene, region):
    """Add shared and per-instance dimensions for a blister surface."""
    label = f"{region.capitalize()} blister"
    width_key = f"{label} width scale"
    lift_key = f"{label} lift scale"

    def factor(label, default, lower, upper, description):
        if label not in ctrl:
            ctrl[label] = float(default)
        ctrl.id_properties_ui(label).update(
            min=lower, max=upper, default=float(default), precision=2, step=0.05,
            description=description)

    def bind(obj, axis, expression, properties):
        obj.driver_remove("scale", axis)
        driver = obj.driver_add("scale", axis).driver
        for name, label in properties.items():
            variable = driver.variables.new()
            variable.name = name
            variable.type = 'SINGLE_PROP'
            variable.targets[0].id = ctrl
            variable.targets[0].data_path = f'["{label}"]'
        driver.expression = expression

    factor(width_key, 1.0, 0.4, 1.6,
           f"Multiply the width of all {region} blisters before applying individual settings.")
    factor(lift_key, 1.0, 0.35, 2.0,
           f"Multiply the crown height of all {region} blisters. Blister amount also applies.")
    by_id = {obj["Defect instance ID"]: obj for obj in objects}
    blisters = [d for d in defects if d["id"] in by_id]
    presets = (1.04, 0.91, 0.96, 1.10, 0.87, 1.0)
    caps = []
    for number, defect in enumerate(blisters, 1):
        obj = by_id[defect["id"]]
        key = f"{label} {number:02d} width"
        default = presets[(number - 1) % len(presets)] if region == "cross-ply" else 1.0
        factor(key, default, 0.4, 1.6,
               f"Width multiplier for {region} blister {number:02d}, relative to the shared width.")
        cap = math.floor(defect["maximum_width_factor"]*1000)/1000
        rib = defect.get("crossed_rib")
        lift_cap = 2.0
        if rib:
            mesh_crown = max(v.co.z for v in obj.data.vertices)
            clearance = rib["top_local_m"]-defect["centre_local_m"][2]-.0008
            lift_cap = min(lift_cap, math.floor(1000*clearance/mesh_crown)/1000)
        obj["Maximum safe width scale"] = cap
        obj["Maximum safe lift scale"] = lift_cap
        obj["Width control"] = key
        defect["scale_controls"] = {"width": width_key, "individual_width": key,
                                    "lift": lift_key, "maximum_width_factor": cap,
                                    "maximum_lift_factor": lift_cap}
        caps.append((obj, key, cap, lift_cap))

    ctrl.update_tag()
    scene.frame_set(scene.frame_current)
    for obj, key, cap, lift_cap in caps:
        for axis in (0, 1):
            bind(obj, axis, f"min({cap:.3f}, max(0.25, shared*individual))",
                 {"shared": width_key, "individual": key})
        bind(obj, 2, f"max(0.001, min({lift_cap:.3f}, enabled*amount*lift))",
             {"enabled": "Enabled", "amount": "Blister amount", "lift": lift_key})
    ctrl.update_tag()
    scene.frame_set(scene.frame_current)


def flange_width_limit(host, origin, boundary):
    """Keep the adjustable footprint on the flat flange and clear of fastener holes."""
    x, y, z = origin
    def supported(scale):
        for fraction in (.25,.5,.75,1.):
            for u,v in boundary:
                dx,dy=u*scale*fraction,v*scale*fraction
                hit, p, n, face = host.ray_cast((x+dx,y+dy,z+.02),(0,0,-1),distance=.025)
                if not hit or abs(p.z-z)>.00015 or n.z<.998 or host.data.polygons[face].material_index!=0:
                    return False
        return True

    # Start slightly inside the requested shape, then expand while the perimeter fits.
    limit = .45
    for step in range(9,53):
        scale = step*.05
        if not supported(scale):
            break
        limit = scale
    return limit


def raised_laminate(tray, clean, ctrl, defects, scene, region="cross-ply"):
    """Create thin surface patches that meet the original laminate under their edges."""
    flange = region == "flange"
    collection_name = FLANGE_BLISTER_COLLECTION if flange else BLISTER_COLLECTION
    material_name = FLANGE_BLISTER_MATERIAL if flange else BLISTER_MATERIAL
    group = bpy.data.collections.get(collection_name) or bpy.data.collections.new(collection_name)
    if group.name not in scene.collection.children:
        scene.collection.children.link(group)
    previous = list(group.objects)
    material = clean.copy()
    material.name = material_name
    tree = material.node_tree
    for ob in tree.nodes:
        if ob.type == 'TEX_COORD':
            ob.object = tray  # The twill and resin grain continue over each lifted patch.
    attribute = tree.nodes.new('ShaderNodeAttribute')
    attribute.attribute_name = FOOTPRINT_ATTRIBUTE
    attribute.label = "Visible footprint of this geometrical blister"
    attribute.location = (1450,-2600)
    label = tree.nodes.new('ShaderNodeOutputAOV')
    label.aov_name = AOVS["blister"]
    label.label = AOVS["blister"]
    label.location = (1740,-2600)
    tree.links.new(attribute.outputs["Fac"],label.inputs["Value"])

    def driver(ob, path, index, expression):
        drv=ob.driver_add(path,index).driver
        for variable,key in (("enabled","Enabled"),("amount","Blister amount")):
            item=drv.variables.new()
            item.name=variable
            item.type='SINGLE_PROP'
            item.targets[0].id=ctrl
            item.targets[0].data_path=f'["{key}"]'
        drv.expression=expression

    host = tray.evaluated_get(bpy.context.evaluated_depsgraph_get()) if flange else None
    for defect in defects:
        if defect["class"]!="blister" or defect["surface"]!=region:
            continue
        rx,ry=defect["radii_m"]
        seed=random.Random(defect["field_seed"])
        phase=[seed.uniform(0,math.tau) for _ in range(5)]
        height=max(.00055,defect["relief_m"]*1.25) if flange else defect["relief_m"]
        edge_depth=.000025 if flange else .00007
        spokes,rings=(64,16) if flange else (80,24)
        boundary=[]
        exponent=seed.uniform(.91,1.14) if flange else seed.uniform(.63,.84)
        for i in range(spokes):
            theta=math.tau*i/spokes
            cx, cy=math.cos(theta), math.sin(theta)
            waviness=(.11*math.sin(2*theta+phase[0]) +
                      .08*math.cos(3*theta+phase[1]) +
                      .042*math.sin(5*theta+phase[2]))
            boundary.append((math.copysign(abs(cx)**exponent,cx)*(1+waviness),
                             math.copysign(abs(cy)**exponent,cy)*(1+waviness)))
        extent_x=max(abs(u) for u,v in boundary)
        extent_y=max(abs(v) for u,v in boundary)
        angle=math.radians(defect["angle_deg"])
        ca,sa=math.cos(angle),math.sin(angle)
        boundary=[(ca*rx*u/extent_x-sa*ry*v/extent_y,sa*rx*u/extent_x+ca*ry*v/extent_y)
                  for u,v in boundary]
        if flange:
            defect["maximum_width_factor"]=flange_width_limit(host,defect["centre_local_m"],boundary)
        points=[(0,0,height-edge_depth)]
        values=[1.0]
        faces=[]
        for ring in range(1,rings+1):
            r=ring/rings
            for i,(u,v) in enumerate(boundary):
                theta=math.tau*i/spokes
                # A wide, slightly slack crown rolls back into the bonded laminate.
                shoulder=(.14 if flange else .33)+.11*math.cos(theta+phase[3])
                t=max(0.,(r-shoulder)/(1.-shoulder))
                envelope=1-t*t*(3-2*t)
                crown=1+.31*r*math.cos(theta+phase[3])+.23*r*r*math.cos(3*theta+phase[4])
                points.append((r*u,r*v,height*envelope*crown-edge_depth))
                values.append(max(0.,min(1.,(1-r)/.14)))
        for i in range(spokes):
            faces.append((0,1+i,1+(i+1)%spokes))
        for band in range(rings-1):
            a=1+band*spokes
            b=a+spokes
            for i in range(spokes):
                j=(i+1)%spokes
                faces.append((a+i,b+i,b+j,a+j))
        rim=1+(rings-1)*spokes
        bottom=len(points)
        points.extend((x,y,-.00036) for x,y,_ in points[rim:])
        values.extend([0.0]*spokes)
        for i in range(spokes):
            j=(i+1)%spokes
            faces.append((rim+i,bottom+i,bottom+j,rim+j))
        faces.append(tuple(reversed(range(bottom,bottom+spokes))))
        data=bpy.data.meshes.new(f"{region.capitalize()} blister profile")
        data.from_pydata(points,[],faces)
        data.update()
        for p in data.polygons:
            p.use_smooth=True
        data.materials.append(material)
        attr=data.attributes.new(FOOTPRINT_ATTRIBUTE,'FLOAT','POINT')
        attr.data.foreach_set("value",values)
        object=bpy.data.objects.new(defect["id"]+(" · raised flange" if flange else " · lifted laminate"),data)
        group.objects.link(object)
        object.parent=tray
        object.location=defect["centre_local_m"]
        object["Defect instance ID"]=defect["id"]
        object["Nominal footprint (mm)"]=[2*rx*1000,2*ry*1000]
        object["Raised crown height (mm)"]=height*1000
        z=defect["centre_local_m"][2]
        driver(object,"location",2,f"{z:.8f}-0.0008*(enabled*amount<=0.001)")
        defect["representation"]=("Irregular raised blister in the moulded flange" if flange else
                                  "Loose laminate lift with irregular shoulders, tapered into the surrounding floor")
        defect["nominal_footprint_mm"]=[2*rx*1000,2*ry*1000]
        defect["mesh_crown_m"]=max(p[2] for p in points)

    for ob in previous:
        old_data=ob.data
        bpy.data.objects.remove(ob,do_unlink=True)
        if old_data and not old_data.users:
            bpy.data.meshes.remove(old_data)
    for old in tuple(bpy.data.materials):
        if old!=material and old.name.startswith(material_name) and not old.users:
            bpy.data.materials.remove(old)
    material.name=material_name
    configure_blister_scale(ctrl,list(group.objects),defects,scene,region)



def generate_anomalies(seed=None, counts=None, scene=None):
    scene=scene or bpy.context.scene
    tray=bpy.data.objects["Battery tray · PA6-GF30"]
    ctrl=anomaly_control(scene)
    seed=int(ctrl["Seed"] if seed is None else seed)
    if not 0<=seed<=2147483647:
        raise ValueError("Seed must be an integer from 0 to 2147483647.")
    requested={k:int(ctrl[label]) for k,label in COUNT_KEYS.items()}
    for key,value in (counts or {}).items():
        if key not in requested or not 0<=int(value)<=24:
            raise ValueError(f"Invalid defect count: {key}={value}")
        requested[key]=int(value)
    defects=sites(tray,seed,requested)
    previous=[tray.data.materials[i] for i in range(2)]
    clean=[bpy.data.materials[name] for name in MATERIALS]
    generated=[anomaly_material(m,region,defects,ctrl) for m,region in zip(clean,("flange","cross-ply"))]
    for i,(m,base) in enumerate(zip(generated,clean)):
        base.use_fake_user=True
        tray.data.materials[i]=m
    for m in previous:
        if m and m.name.startswith("Tray anomalies ·") and not m.users:
            bpy.data.materials.remove(m)
    for mat,region in zip(generated,("flange","cross-ply")):
        mat.name="Tray anomalies · "+region
    raised_laminate(tray,clean[1],ctrl,defects,scene)
    raised_laminate(tray,clean[0],ctrl,defects,scene,region="flange")
    ctrl["Seed"]=seed
    for key,n in requested.items():
        ctrl[COUNT_KEYS[key]]=n
    for layer in scene.view_layers:
        for name in AOVS.values():
            if not layer.aovs.get(name):
                item=layer.aovs.add()
                item.name=name
                item.type='VALUE'
    manifest=bpy.data.texts.get(MANIFEST) or bpy.data.texts.new(MANIFEST)
    manifest.clear()
    manifest.write(json.dumps({"seed":seed, "units":"metres, tray-local coordinates",
                               "class_mask_passes":AOVS, "instances":defects},indent=2))
    scene["Anomaly generator"]="generate_tray_anomalies.py"
    scene["Anomaly labels"]="Per-instance metadata in anomaly_instances.json; three per-class shader AOV masks."
    ctrl.update_tag()
    scene.frame_set(scene.frame_current)
    return defects


if __name__ == "__main__":
    generate_anomalies()
