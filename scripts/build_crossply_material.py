def build_crossply_material(name):
    """Build the procedural cross-ply finish using tray-local metres."""
    import bpy
    import math

    material = bpy.data.materials.new(name)
    material.diffuse_color = (0.014, 0.018, 0.023, 1)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    def node(kind, label, x, y):
        item = tree.nodes.new(kind)
        item.label = label
        item.location = (x, y)
        return item

    def connect(value, socket):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value

    def math_node(operation, a, b, label, x, y, clamp=False):
        item = node("ShaderNodeMath", label, x, y)
        item.operation = operation
        item.use_clamp = clamp
        connect(a, item.inputs[0])
        if b is not None:
            connect(b, item.inputs[1])
        return item.outputs[0]

    def ease(value, points, label, x, y):
        item = node("ShaderNodeValToRGB", label, x, y)
        item.color_ramp.interpolation = 'EASE'
        for element, (position, scalar) in zip(item.color_ramp.elements, points):
            element.position = position
            element.color = (scalar, scalar, scalar, 1)
        connect(value, item.inputs["Fac"])
        return item.outputs["Color"]

    def noise(vector, scale, detail, roughness, label, x, y):
        item = node("ShaderNodeTexNoise", label, x, y)
        item.inputs["Scale"].default_value = scale
        item.inputs["Detail"].default_value = detail
        item.inputs["Roughness"].default_value = roughness
        connect(vector, item.inputs["Vector"])
        return item.outputs["Fac"]

    def bump(height, strength, distance, normal, label, x, y):
        item = node("ShaderNodeBump", label, x, y)
        item.inputs["Strength"].default_value = strength
        item.inputs["Distance"].default_value = distance
        connect(height, item.inputs["Height"])
        if normal is not None:
            connect(normal, item.inputs["Normal"])
        return item.outputs["Normal"]

    coordinates = node("ShaderNodeTexCoord", "Object coordinates · metres", -2120, 110)
    uvw = coordinates.outputs["Object"]
    axes = node("ShaderNodeSeparateXYZ", "Tray X and Y fibre axes", -1910, 850)
    connect(uvw, axes.inputs[0])
    scale = 1 / 0.0006
    x = math_node('MULTIPLY', axes.outputs["X"], scale, "600 µm pitch", -1720, 1060)
    y = math_node('MULTIPLY', axes.outputs["Y"], scale, "600 µm pitch", -1720, 620)

    # The two yarns remain at 0° and 90°. Their alternating crossover produces the diagonal.
    def yarn(across, along, flip, row):
        cross_fraction = math_node('FRACT', across, None, "Across-yarn position", -1520, row)
        radians = math_node('MULTIPLY', cross_fraction, math.pi, "Rounded tow profile", -1340, row)
        sine = math_node('SINE', radians, None, "Tow crown", -1160, row)
        positive = math_node('MAXIMUM', sine, 0.0, "Nonnegative crown", -980, row)
        crown = math_node('POWER', positive, 1.6, "Resin-rounded tow", -800, row)
        pick = math_node('FLOOR', across, None, "Pick index", -1520, row-205)
        diagonal = math_node('ADD', along, pick, "Advance one pick per row", -1340, row-205)
        phase = math_node('MULTIPLY', diagonal, math.pi/2, "2 over, 2 under", -1160, row-205)
        cycle = math_node('SINE', phase, None, "Weave crossover", -980, row-205)
        amplitude = math_node('MULTIPLY', cycle, -0.33 if flip else 0.33, "Over-under height", -800, row-205)
        lift = math_node('ADD', amplitude, 0.67, "Tow lift", -620, row-205)
        return math_node('MULTIPLY', crown, lift, "Woven tow height", -430, row)

    warp = yarn(y, x, False, 1080)
    weft = yarn(x, y, True, 630)
    weave = math_node('MAXIMUM', warp, weft, "Diagonal twill print-through", -220, 840)
    twill_normal = bump(weave, 0.25, 0.000075, None, "Twill · 0.6 mm pitch", 30, 840)

    grain = node("ShaderNodeTexVoronoi", "Rounded pebbles · 1.1 mm", -1730, 110)
    grain.voronoi_dimensions = '2D'
    grain.feature = 'SMOOTH_F1'
    grain.inputs["Scale"].default_value = 1 / 0.0011
    grain.inputs["Smoothness"].default_value = 0.53
    connect(uvw, grain.inputs["Vector"])
    distance = grain.outputs["Distance"]
    peaks = ease(distance, [(0.10, 1.0), (0.67, 0.0)], "Rounded pebble crowns", -1430, 190)
    patches = noise(uvw, 7, 4, 0.55, "Uneven pebble patches", -1730, -220)
    amplitude = ease(patches, [(0.30, 0.55), (0.70, 1.0)], "Patch-to-patch pebble strength", -1430, -95)
    coverage = ease(patches, [(0.44, 0.0), (0.62, 1.0)], "Feathered patches and bare resin", -1430, -380)
    patch_gain = math_node('MULTIPLY', amplitude, coverage, "Gated pebbles", -1110, -100)
    relief = math_node('MULTIPLY', patch_gain, 0.55, "Patch relief variation", -1090, 115)
    relief = math_node('ADD', relief, 0.45, "Continuous pebble relief under clear resin", -900, 115)
    pebble = math_node('MULTIPLY', peaks, relief, "Rounded pebbles across the laminate", -710, 150)
    pebble_normal = bump(pebble, 0.45, 0.0005, twill_normal, "Pebble bump · 0.50 mm distance", 250, 650)

    rough_signal = noise(uvw, 6, 6, 0.6, "Resin roughness variation", -1730, -725)
    roughness = ease(rough_signal, [(0.32, 0.12), (0.68, 0.32)], "Glossy black resin", -1430, -725)
    haze_peak = ease(distance, [(0.25, 1.0), (0.65, 0.15)], "Frost on pebble peaks", -1430, -1020)
    edge = math_node('MULTIPLY', haze_peak, 4.0, "Peak mask ×4, clamped", -1110, -1020, True)
    haze_patch = math_node('MULTIPLY', edge, coverage, "Haze confined to pebble patches", -890, -1020)
    haze_amount = math_node('MULTIPLY', haze_patch, 0.24, "Reduced haze strength · 0.24", -670, -1020)

    # Equal-pitch profiles on both tray axes provide a coarser, regular cross-ply texture.
    def course(coordinate, name, row, spacing=0.0024):
        pitch = math_node('MULTIPLY', coordinate, 1 / spacing, name + f" · {spacing * 1000:g} mm pitch", -360, row)
        phase = math_node('MULTIPLY', pitch, 2 * math.pi, "Periodic strand profile", -170, row)
        crest = math_node('COSINE', phase, None, "Rounded strand crest", 20, row)
        return ease(crest, [(0.0, 0.0), (1.0, 1.0)], "Soft course edges", 210, row)

    along_x = course(axes.outputs["Y"], "X-direction fibre courses", -475)
    along_y = course(axes.outputs["X"], "Y-direction fibre courses", -730)
    crossed = math_node('ADD', along_x, along_y, "Balanced 0° + 90° courses", 510, -460)
    crossed = math_node('MULTIPLY', crossed, 0.5, "Equal weight on both axes", 700, -460)
    course_normal = bump(crossed, 0.38, 0.00025, pebble_normal,
                         "Coarser cross-ply courses", 905, 650)
    micrograin = noise(uvw, 1450, 2.0, 0.61, "Fine resin granularity", -360, -1010)
    grain_normal = bump(micrograin, 0.28, 0.000125, course_normal,
                        "Fine grain over both fibre directions", 1125, 650)

    # A shallow broader fibre band stays visible through lighting and image downsampling.
    broad_x = course(axes.outputs["Y"], "Broad X fibre bands", -1360, spacing=0.0058)
    broad_y = course(axes.outputs["X"], "Broad Y fibre bands", -1610, spacing=0.0058)
    broad = math_node('ADD', broad_x, broad_y, "Crossed band orientations", 520, -1360)
    broad = math_node('MULTIPLY', broad, 0.5, "Equal band contribution", 720, -1360)
    broad_normal = bump(broad, 0.26, 0.00034, grain_normal, "Shallow broad fibre relief", 1355, 655)

    def band_colour(lo, hi, label, y):
        item = node("ShaderNodeValToRGB", label, 975, y)
        item.color_ramp.interpolation = 'EASE'
        for element, colour in zip(item.color_ramp.elements, (lo, hi)):
            element.color = (*colour, 1)
        connect(broad, item.inputs["Fac"])
        return item.outputs["Color"]

    resin_colour = band_colour((.0029, .0036, .0046), (.0066, .0080, .0102), "Mild dark resin band contrast", -1180)
    haze_colour = band_colour((.165, .182, .204), (.210, .232, .258), "Fibre bands visible through haze", -1500)
    rough_swing = math_node('MULTIPLY', broad, .092, "Broad band gloss variation", 990, -1830)
    rough_swing = math_node('SUBTRACT', rough_swing, .029, "Balance the roughness change", 1190, -1830)
    resin_roughness = math_node('ADD', roughness, rough_swing, "Panel roughness plus crossed bands", 1380, -1830)
    haze_roughness = math_node('ADD', 0.61, rough_swing, "Haze roughness plus crossed bands", 1380, -2060)

    future = node("ShaderNodeValue", "Optional blister height · zero by default", 1360, 110)
    future.outputs[0].default_value = 0.0
    surface = bump(future.outputs[0], 1.0, 0.001, broad_normal,
                   "Optional dome normal over the same surface texture", 1570, 650)
    resin = node("ShaderNodeBsdfPrincipled", "Wet-black PA6 resin", 1775, 750)
    connect(resin_colour, resin.inputs["Base Color"])
    resin.inputs["IOR"].default_value = 1.53
    resin.inputs["Specular IOR Level"].default_value = 0.44
    resin.inputs["Coat Weight"].default_value = 0.16
    resin.inputs["Coat Roughness"].default_value = 0.17
    connect(resin_roughness, resin.inputs["Roughness"])
    connect(surface, resin.inputs["Normal"])
    connect(surface, resin.inputs["Coat Normal"])

    haze = node("ShaderNodeBsdfPrincipled", "Rough, whitish resin haze", 1775, -35)
    connect(haze_colour, haze.inputs["Base Color"])
    connect(haze_roughness, haze.inputs["Roughness"])
    haze.inputs["IOR"].default_value = 1.49
    haze.inputs["Specular IOR Level"].default_value = 0.24
    connect(surface, haze.inputs["Normal"])

    mix = node("ShaderNodeMixShader", "Selective frosty haze over glossy resin", 2120, 340)
    connect(haze_amount, mix.inputs[0])
    connect(resin.outputs[0], mix.inputs[1])
    connect(haze.outputs[0], mix.inputs[2])
    output = node("ShaderNodeOutputMaterial", "Cross-ply finish", 2350, 340)
    connect(mix.outputs[0], output.inputs["Surface"])

    material["Appearance basis"] = "Visual cross-ply recipe from the user references. 0°/90° yarn axes, diagonal twill print-through, patchy pebbles and frosty haze."
    material["Twill pitch (mm)"] = 0.6
    material["Pebble nominal size (mm)"] = 1.1
    material["Biaxial fibre course pitch (mm)"] = 2.4
    material["Haze strength"] = 0.24
    material["Broad cross-ply band pitch (mm)"] = 5.8
    return material
