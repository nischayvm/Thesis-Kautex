"""Controls and four-camera capture for the ANNAlyzer inspection scene."""

from datetime import datetime
from math import pi
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp

import bpy


bl_info = {
    "name": "ANNAlyzer Controls",
    "author": "ANNAlyzer",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "3D View > Sidebar > ANNAlyzer",
    "category": "3D View",
}

LIGHTING = "Lighting levels · select for custom properties"
CAPTURE = "Capture controls"
BARS = ("Front", "Rear", "Left end", "Right end")
CAMERAS = ("Front left", "Front right", "Rear left", "Rear right")
RESPONSE = (
    ("Diffuse response", "diffuse_factor"),
    ("Specular response", "specular_factor"),
    ("Transmission response", "transmission_factor"),
    ("Volume response", "volume_factor"),
)
EXPOSURE = "Exposure offset (EV)"
WHITE_BALANCE = "Override white balance"
TEMPERATURE = "White balance (K)"
TINT = "White balance tint"
DIRECTORY = "Capture folder"
PNG_16 = "Capture 16-bit PNG"
CAPTURE_JOB = None
CAPTURE_STATUS = ""


def setting(owner, name, default, description="", **limits):
    if name not in owner:
        owner[name] = default
    owner.id_properties_ui(name).update(description=description, **limits)


def prop(layout, owner, name, label=None):
    layout.prop(owner, f'["{name}"]', text=label or name)


def link_driver(owner, path, expression, variables, index=None):
    if index is None:
        owner.driver_remove(path)
        channel = owner.driver_add(path)
    else:
        owner.driver_remove(path, index)
        channel = owner.driver_add(path, index)
    driver = channel.driver
    for label, (target, source) in variables.items():
        variable = driver.variables.new()
        variable.name = label
        variable.type = 'SINGLE_PROP'
        variable.targets[0].id = target
        variable.targets[0].data_path = source
    driver.expression = expression


def custom(owner, name, index=None):
    path = f'["{name}"]'
    return owner, path if index is None else f'{path}[{index}]'


def fixture_color(levels, group, bar, axis):
    variables = {
        'rgb': custom(levels, f'{group} RGB', axis),
        'kelvin_on': custom(levels, f'{group} Use Kelvin'),
    }
    mode = 'kelvin_on'
    if bar:
        variables['tint'] = custom(levels, f'{bar} bar RGB tint', axis)
        variables['override'] = custom(levels, f'{bar} bar Override Kelvin')
        mode = 'max(kelvin_on, override)'
    expression = f'(1 if {mode} else rgb)' + ('*tint' if bar else '')
    return expression, variables


def fixture_kelvin(levels, group, bar):
    variables = {'group_k': custom(levels, f'{group} Kelvin')}
    if not bar:
        return 'group_k', variables
    variables.update(override=custom(levels, f'{bar} bar Override Kelvin'),
                     bar_k=custom(levels, f'{bar} bar Kelvin'))
    return 'bar_k if override else group_k', variables


def configure_fixture(light, levels, motion, group, bar):
    for axis in range(3):
        link_driver(light, 'color', *fixture_color(levels, group, bar, axis), index=axis)
    kelvin_expression, kelvin_variables = fixture_kelvin(levels, group, bar)
    link_driver(light, 'temperature', kelvin_expression, kelvin_variables)
    _, color_variables = fixture_color(levels, group, bar, 0)
    mode = 'max(kelvin_on, override)' if bar else 'kelvin_on'
    link_driver(light, 'use_temperature', mode,
                {k: v for k,v in color_variables.items() if k in {'kelvin_on','override'}})
    beam = {'beam': custom(levels, f'{group} beam spread (deg)')}
    expression = 'beam'
    if bar:
        beam.update(dark=custom(motion, 'Dark-field mix'), grazing=custom(motion, 'Grazing beam width (deg)'),
                    override=custom(levels, f'{bar} bar Override Beam'),
                    bar_beam=custom(levels, f'{bar} bar beam spread (deg)'))
        expression = '(bar_beam if override else beam)*(1-dark)+grazing*dark'
    link_driver(light, 'spread', f'({expression})*{pi/180}', beam)
    for label, attr in RESPONSE + (('Shadows', 'use_shadow'),):
        link_driver(light, attr, 'value', {'value': custom(levels, f'{group} {label}')})
    for axis in ('size', 'size_y'):
        base = f'ANNAlyzer native {axis}'
        if base not in light:
            light[base] = getattr(light, axis)
        link_driver(light, axis, f'{light[base]:.12g}*scale',
                    {'scale': custom(levels, f'{group} emitter size scale')})


def configure_diffuser(material, levels, group, bar):
    tree = material.node_tree
    bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    color = bsdf.inputs['Emission Color']
    if color.is_linked and color.links[0].from_node.label == 'ANNAlyzer LED emission':
        return
    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.label = 'LED RGB and tint'
    rgb.location = (-850, -380)
    for axis in range(3):
        link_driver(rgb.outputs[0], 'default_value', *fixture_color(levels, group, bar, axis), index=axis)
    blackbody = tree.nodes.new('ShaderNodeBlackbody')
    blackbody.location = (-850, -560)
    link_driver(blackbody.inputs['Temperature'], 'default_value', *fixture_kelvin(levels, group, bar))
    blend = tree.nodes.new('ShaderNodeMixRGB')
    blend.label = 'LED temperature mode'
    blend.location = (-600, -510)
    _, color_variables = fixture_color(levels, group, bar, 0)
    link_driver(blend.inputs[0], 'default_value', 'max(kelvin_on, override)' if bar else 'kelvin_on',
                {k: v for k,v in color_variables.items() if k in {'kelvin_on','override'}})
    blend.inputs[1].default_value = (1, 1, 1, 1)
    tree.links.new(blackbody.outputs[0], blend.inputs[2])
    output = tree.nodes.new('ShaderNodeMixRGB')
    output.label = 'ANNAlyzer LED emission'
    output.location = (-340, -390)
    output.blend_type = 'MULTIPLY'
    output.inputs[0].default_value = 1.0
    tree.links.new(rgb.outputs[0], output.inputs[1])
    tree.links.new(blend.outputs[0], output.inputs[2])
    tree.links.new(output.outputs[0], color)


def setup_scene(scene):
    """Wire the project fixtures and initialize editable settings in the blend file."""
    levels, motion = scene.objects[LIGHTING], scene.objects[CAPTURE]
    fixture_groups = {
        'Ceiling': (bpy.data.collections['Ceiling LED fixtures'], None, (1., .91, .80)),
        'Rig': (bpy.data.collections['Rig task lighting'], BARS, (1., .94, .88)),
    }
    for group, (_, bars, rgb) in fixture_groups.items():
        setting(levels, f'{group} RGB', rgb, "Light color in RGB mode.", subtype='COLOR', min=0., max=1.)
        setting(levels, f'{group} Use Kelvin', False, "Use correlated color temperature.")
        setting(levels, f'{group} Kelvin', 6500., "White LED color temperature.", min=1000., max=20000.)
        setting(levels, f'{group} beam spread (deg)', 180., "Full beam angle in degrees.", min=1., max=180.)
        setting(levels, f'{group} emitter size scale', 1., "Scale the effective light source for softer or harder shadows.",
                min=.1, max=4., soft_max=2.)
        setting(levels, f'{group} Shadows', True, "Allow fixtures to cast shadows.")
        for label, _ in RESPONSE:
            setting(levels, f'{group} {label}', 1., "Contribution multiplier.", min=0., max=1.)
        for bar in bars or ():
            setting(levels, f'{bar} bar RGB tint', (1.,1.,1.), "Color multiplier for this bar in either color mode.",
                    subtype='COLOR', min=0., max=1.)
            setting(levels, f'{bar} bar Override Kelvin', False, "Use a separate Kelvin temperature for this bar.")
            setting(levels, f'{bar} bar Kelvin', 6500., "Individual LED color temperature.", min=1000., max=20000.)
            setting(levels, f'{bar} bar Override Beam', False, "Use this bar's beam spread outside dark-field mode.")
            setting(levels, f'{bar} bar beam spread (deg)', 180., "Full beam angle in degrees.", min=1., max=180.)
    for group, (collection, bars, _) in fixture_groups.items():
        for obj in collection.objects:
            if obj.type == 'LIGHT':
                bar = obj.name.split(' · ', 1)[1] if bars else None
                configure_fixture(obj.data, levels, motion, group, bar)
        if bars:
            for bar in bars:
                configure_diffuser(bpy.data.materials[f'{bar} bar · opal diffuser'], levels, group, bar)
        else:
            configure_diffuser(bpy.data.materials['Ceiling LED · opal diffuser'], levels, group, None)
    background = next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    setting(levels, 'World fill strength', float(background.inputs['Strength'].default_value),
            "Environment contribution. Use zero for a dark inspection cell.", min=0., max=4., soft_max=1.)
    link_driver(background.inputs['Strength'], 'default_value', 'value',
                {'value': custom(levels, 'World fill strength')})
    setting(motion, DIRECTORY, '//../data', "Blender-relative or absolute directory. Each capture creates a unique subfolder.")
    setting(motion, PNG_16, False, "Write 16-bit RGB PNGs instead of 8-bit RGB PNGs.")
    for name in CAMERAS:
        camera = scene.objects[f'Inspect · {name}']
        setting(camera, EXPOSURE, 0., "Stops added to the scene exposure at capture time.", min=-8., max=8.)
        setting(camera, WHITE_BALANCE, False, "Use a separate white balance for this camera at capture time.")
        setting(camera, TEMPERATURE, 6500., "Per-camera white balance temperature.", min=1800., max=18000.)
        setting(camera, TINT, 10., "Per-camera green/magenta adjustment.", min=-150., max=150.)
    levels.update_tag()
    motion.update_tag()
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)


def capture_folder(scene):
    base = Path(bpy.path.abspath(scene.objects[CAPTURE][DIRECTORY])).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=datetime.now().strftime('capture_%Y%m%d_%H%M%S_'), dir=base))


def capture_four(scene, folder=None):
    """Render four display-referred PNGs and restore the artist's scene settings."""
    motion = scene.objects[CAPTURE]
    cameras = [(name, scene.objects[f'Inspect · {name}']) for name in CAMERAS]
    folder = capture_folder(scene) if folder is None else Path(folder)
    render, view = scene.render, scene.view_settings
    image = render.image_settings
    view_keys = ('exposure','use_white_balance','white_balance_temperature','white_balance_tint')
    image_keys = ('file_format','color_mode','color_depth','color_management')
    render_keys = ('filepath','use_file_extension','use_multiview')
    saved = [(owner, {k:getattr(owner,k) for k in keys})
             for owner, keys in ((view,view_keys),(image,image_keys),(render,render_keys))]
    old_camera = scene.camera
    base_view = saved[0][1]
    try:
        image.file_format = 'PNG'
        image.color_mode = 'RGB'
        image.color_depth = '16' if motion[PNG_16] else '8'
        image.color_management = 'FOLLOW_SCENE'
        render.use_file_extension = True
        render.use_multiview = False
        for index,(name,camera) in enumerate(cameras,1):
            scene.camera = camera
            view.exposure = base_view['exposure']+camera[EXPOSURE]
            override = camera[WHITE_BALANCE]
            view.use_white_balance = bool(override or base_view['use_white_balance'])
            view.white_balance_temperature = camera[TEMPERATURE] if override else base_view['white_balance_temperature']
            view.white_balance_tint = camera[TINT] if override else base_view['white_balance_tint']
            output = folder / (name.lower().replace(' ','_')+'.png')
            render.filepath = str(output)
            print(f'ANNAlyzer capture {index}/4: {output}', flush=True)
            result = bpy.ops.render.render(write_still=True, scene=scene.name)
            if result != {'FINISHED'} or not output.is_file():
                raise RuntimeError(f'Capture interrupted at {name}. Output so far: {folder}')
    finally:
        scene.camera = old_camera
        for owner, state in saved:
            for key, value in state.items():
                setattr(owner,key,value)
    return folder


def redraw_capture_panel():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def poll_capture_job():
    global CAPTURE_JOB, CAPTURE_STATUS
    if CAPTURE_JOB is None:
        return None
    process, work, folder, log_path = CAPTURE_JOB
    code = process.poll()
    count = sum((folder / (name.lower().replace(' ','_')+'.png')).is_file() for name in CAMERAS)
    if code is None:
        CAPTURE_STATUS = f'Rendering in background · {count}/4 images'
        redraw_capture_panel()
        return .75
    if code == 0 and count == 4:
        CAPTURE_STATUS = 'Saved all four images'
        print(f'ANNAlyzer capture complete: {folder}')
        shutil.rmtree(work, ignore_errors=True)
    else:
        CAPTURE_STATUS = f'Capture failed · {count}/4 images. See System Console.'
        print(f'ANNAlyzer capture failed. Output: {folder}. Render log: {log_path}', flush=True)
        print(log_path.read_text(encoding='utf-8', errors='replace')[-8000:], flush=True)
    CAPTURE_JOB = None
    redraw_capture_panel()
    return None


def start_capture_job(scene):
    global CAPTURE_JOB, CAPTURE_STATUS
    if CAPTURE_JOB is not None:
        raise RuntimeError('A four-camera capture is already running.')
    if not bpy.data.filepath:
        raise RuntimeError('Save the Blender scene before using background capture.')
    work = Path(mkdtemp(prefix='annalyzer_render_'))
    snapshot = work / 'inspection_snapshot.blend'
    log_path = work / 'render.log'
    folder = None
    try:
        folder = capture_folder(scene)
        result = bpy.ops.wm.save_as_mainfile(filepath=str(snapshot), copy=True, relative_remap=True)
        if result != {'FINISHED'} or not snapshot.is_file():
            raise RuntimeError('Could not save a temporary snapshot for the render.')
        arguments = [bpy.app.binary_path, '--background', str(snapshot), '--python-exit-code', '1',
                     '--python', str(Path(__file__).resolve()), '--',
                     '--annalyzer-worker', scene.name, str(folder)]
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        with log_path.open('wb') as log:
            process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT, **options)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        if folder is not None:
            folder.rmdir()
        raise
    CAPTURE_JOB = (process, work, folder, log_path)
    CAPTURE_STATUS = 'Rendering in background · 0/4 images'
    bpy.app.timers.register(poll_capture_job, first_interval=.5, persistent=True)
    redraw_capture_panel()
    return folder


class ANNALYZER_OT_render_four(bpy.types.Operator):
    bl_idname = 'annalyzer.render_four'
    bl_label = 'Render All 4 Cameras'
    bl_description = 'Render a snapshot of the current frame from four cameras in a background Blender process'

    def execute(self, context):
        try:
            folder = start_capture_job(context.scene)
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Background capture started. Output: {folder}')
        return {'FINISHED'}


class ANNALYZER_OT_view_camera(bpy.types.Operator):
    bl_idname = 'annalyzer.view_camera'
    bl_label = 'View Selected Camera'
    bl_description = 'Look through the selected inspection camera'

    def execute(self, context):
        context.scene.camera = context.scene.objects[f'Inspect · {context.scene.annalyzer_camera}']
        context.space_data.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class ANNAlyzerPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ANNAlyzer'

    @classmethod
    def poll(cls, context):
        motion = context.scene.objects.get(CAPTURE)
        return motion is not None and DIRECTORY in motion and LIGHTING in context.scene.objects


class ANNALYZER_PT_capture(ANNAlyzerPanel, bpy.types.Panel):
    bl_label = 'Inspection Capture'

    def draw(self, context):
        scene, layout = context.scene, self.layout
        row = layout.row()
        row.enabled = CAPTURE_JOB is None
        row.operator('annalyzer.render_four', icon='RENDER_STILL')
        if CAPTURE_STATUS:
            layout.label(text=CAPTURE_STATUS)
        prop(layout,scene.objects[CAPTURE],DIRECTORY,'Output Folder')
        prop(layout,scene.objects[CAPTURE],PNG_16,'16-bit PNG')


class ANNALYZER_PT_lighting(ANNAlyzerPanel, bpy.types.Panel):
    bl_label = 'LED Lighting'
    bl_parent_id = 'ANNALYZER_PT_capture'

    def draw(self, context):
        scene, layout = context.scene, self.layout
        levels, motion = scene.objects[LIGHTING], scene.objects[CAPTURE]
        prop(layout,levels,'Ceiling LEDs','Ceiling Intensity')
        prop(layout,levels,'Rig LEDs','Rig Intensity')
        prop(layout,motion,'All 4 light height (m)','All Bar Heights (m)')
        prop(layout,motion,'Dark-field mix','Dark-Field Mix')
        if motion['Dark-field mix'] > 0.:
            for label in ('Grazing height (m)','Grazing down angle (deg)',
                          'Grazing beam width (deg)','Ceiling fraction in dark field'):
                prop(layout,motion,label)
        prop(layout,levels,'World fill strength','World Fill')
        layout.separator()
        layout.prop(scene,'annalyzer_led')
        choice = scene.annalyzer_led
        if choice in ('Ceiling','Rig'):
            prop(layout,levels,f'{choice} Use Kelvin','Use Kelvin')
            if levels[f'{choice} Use Kelvin']:
                prop(layout,levels,f'{choice} Kelvin','Temperature (K)')
            else:
                prop(layout,levels,f'{choice} RGB','LED Color')
            prop(layout,levels,f'{choice} beam spread (deg)','Beam Spread (°)')
            prop(layout,levels,f'{choice} emitter size scale','Emitter Size Scale')
            prop(layout,levels,f'{choice} Shadows','Shadows')
            for label,_ in RESPONSE:
                prop(layout,levels,f'{choice} {label}',label)
        else:
            for suffix,label in (('intensity','Bar Intensity'),('height (m)','Bar Height (m)'),
                                 ('down angle (deg)','Bar Down Angle (°)')):
                prop(layout,motion,f'{choice} bar {suffix}',label)
            prop(layout,levels,f'{choice} bar RGB tint','Bar RGB Tint')
            prop(layout,levels,f'{choice} bar Override Kelvin','Override Kelvin')
            if levels[f'{choice} bar Override Kelvin']:
                prop(layout,levels,f'{choice} bar Kelvin','Temperature (K)')
            prop(layout,levels,f'{choice} bar Override Beam','Override Beam Spread')
            if levels[f'{choice} bar Override Beam']:
                prop(layout,levels,f'{choice} bar beam spread (deg)','Beam Spread (°)')


class ANNALYZER_PT_camera(ANNAlyzerPanel, bpy.types.Panel):
    bl_label = 'Inspection Cameras'
    bl_parent_id = 'ANNALYZER_PT_capture'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        scene, layout = context.scene, self.layout
        name = scene.annalyzer_camera
        camera = scene.objects[f'Inspect · {name}']
        data, motion = camera.data, scene.objects[CAPTURE]
        layout.prop(scene,'annalyzer_camera')
        layout.operator('annalyzer.view_camera',icon='CAMERA_DATA')
        prop(layout,motion,'All 4 camera height (m)','All Camera Heights (m)')
        prop(layout,motion,f'{name} camera height (m)','Camera Height (m)')
        target = scene.objects.get(f'Aim · {name}')
        if target is not None:
            layout.prop(target,'location',text='Aim Target')
        layout.prop(data,'lens',text='Focal Length')
        layout.prop(data,'sensor_fit',text='Sensor Fit')
        if data.sensor_fit != 'VERTICAL':
            layout.prop(data,'sensor_width')
        else:
            layout.prop(data,'sensor_height')
        row=layout.row(align=True)
        row.prop(data,'shift_x')
        row.prop(data,'shift_y')
        row=layout.row(align=True)
        row.prop(data,'clip_start')
        row.prop(data,'clip_end')
        layout.prop(data.dof,'use_dof',text='Depth of Field')
        if data.dof.use_dof:
            layout.prop(data.dof,'focus_object')
            if data.dof.focus_object is None:
                layout.prop(data.dof,'focus_distance')
            layout.prop(data.dof,'aperture_fstop',text='Aperture f-stop')
            layout.prop(data.dof,'aperture_blades',text='Aperture Blades')
            layout.prop(data.dof,'aperture_rotation',text='Aperture Rotation')
            layout.prop(data.dof,'aperture_ratio',text='Aperture Ratio')
        prop(layout,camera,EXPOSURE,'Capture Exposure Offset (EV)')
        prop(layout,camera,WHITE_BALANCE,'Override Capture White Balance')
        if camera[WHITE_BALANCE]:
            prop(layout,camera,TEMPERATURE,'White Balance (K)')
            prop(layout,camera,TINT)


class ANNALYZER_PT_image(ANNAlyzerPanel, bpy.types.Panel):
    bl_label = 'Image and Render Settings'
    bl_parent_id = 'ANNALYZER_PT_capture'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        scene, layout = context.scene, self.layout
        view, render = scene.view_settings, scene.render
        layout.prop(view,'exposure',text='Base Exposure (EV)')
        layout.prop(view,'view_transform')
        layout.prop(view,'look')
        layout.prop(view,'gamma')
        layout.prop(view,'use_white_balance',text='Use Scene White Balance')
        if view.use_white_balance:
            layout.prop(view,'white_balance_temperature',text='White Balance (K)')
            layout.prop(view,'white_balance_tint')
        row=layout.row(align=True)
        row.prop(render,'resolution_x',text='Width')
        row.prop(render,'resolution_y',text='Height')
        layout.prop(render,'resolution_percentage',text='Resolution Scale')
        row=layout.row(align=True)
        row.prop(render,'pixel_aspect_x')
        row.prop(render,'pixel_aspect_y')
        if render.engine == 'CYCLES':
            layout.prop(scene.cycles,'device')
            layout.prop(scene.cycles,'samples')
            layout.prop(scene.cycles,'use_denoising')
        layout.prop(render,'use_motion_blur')
        if render.use_motion_blur:
            layout.prop(render,'motion_blur_shutter',text='Shutter (Frames)')


CLASSES = (ANNALYZER_OT_render_four,ANNALYZER_OT_view_camera,ANNALYZER_PT_capture,
           ANNALYZER_PT_lighting,ANNALYZER_PT_camera,ANNALYZER_PT_image)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.annalyzer_camera = bpy.props.EnumProperty(
        name='Camera', items=[(name,name,'') for name in CAMERAS])
    bpy.types.Scene.annalyzer_led = bpy.props.EnumProperty(
        name='Fixture Group or Bar', items=[(name,name,'') for name in ('Ceiling','Rig',*BARS)])


def unregister():
    del bpy.types.Scene.annalyzer_led
    del bpy.types.Scene.annalyzer_camera
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__' and '--annalyzer-worker' in sys.argv:
    argument_index = sys.argv.index('--annalyzer-worker')
    scene_name, directory = sys.argv[argument_index+1:argument_index+3]
    capture_four(bpy.data.scenes[scene_name], directory)
