"""Run in background Blender with -- --output DIRECTORY [--count 100]."""

import argparse
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import time

import bpy
import numpy as np


CAMERA = 'Inspect · Front left'
SIDE = 2048
BASE_SEED = 24117


def normal_and_mask_nodes(scene, folder):
    tree = bpy.data.node_groups.new('Prototype binary blister mask', 'CompositorNodeTree')
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    image = tree.nodes.new('NodeGroupOutput')
    layer = tree.nodes.new('CompositorNodeRLayers')
    tree.links.new(layer.outputs['Image'], image.inputs['Image'])
    binary = tree.nodes.new('ShaderNodeMath')
    binary.operation = 'GREATER_THAN'
    binary.inputs[1].default_value = .1
    tree.links.new(layer.outputs['Defect_Blister'], binary.inputs[0])
    save = tree.nodes.new('CompositorNodeOutputFile')
    save.directory = str(folder)
    save.file_output_items.new('FLOAT', 'Mask')
    save.format.media_type = 'IMAGE'
    save.format.file_format = 'PNG'
    save.format.color_mode = 'BW'
    save.format.color_depth = '8'
    save.format.color_management = 'OVERRIDE'
    save.format.view_settings.view_transform = 'Raw'
    save.format.view_settings.exposure = 0.
    save.format.view_settings.gamma = 1.
    save.format.view_settings.use_white_balance = False
    tree.links.new(binary.outputs[0], save.inputs['Mask'])
    return save


def png_header(path):
    with path.open('rb') as image:
        header = image.read(26)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or len(header) < 26:
        raise ValueError(f'Invalid PNG: {path}')
    return (*struct.unpack('>II', header[16:24]), header[24], header[25])


def positive_pixels(path, side=SIDE):
    if png_header(path) != (side, side, 8, 0):
        raise ValueError(f'Unexpected mask format: {path}')
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        data = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(data)
        channel = data[::4]
        on = int(np.count_nonzero(channel == 1.))
        if not 0 < on < channel.size or np.count_nonzero(channel) != on:
            raise ValueError(f'Empty or nonbinary blister mask: {path}')
        return on
    finally:
        bpy.data.images.remove(image)


def render(scene, destination, side=SIDE):
    scene.render.filepath = str(destination)
    if bpy.ops.render.render(write_still=True, scene=scene.name) != {'FINISHED'}:
        raise RuntimeError(f'Render failed: {destination}')
    if png_header(destination) != (side, side, 8, 2):
        raise ValueError(f'Unexpected image format: {destination}')


def run(folder, count, base_seed):
    start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {folder}')
    normal, defective, masks, temporary = (folder / n for n in ('normal','bubbles','masks','.mask_render'))
    for directory in (normal, defective, masks, temporary):
        directory.mkdir()

    scene = bpy.context.scene
    camera = scene.objects[CAMERA]
    scene.camera = camera
    scene.frame_set(1)
    scene.camera = camera
    render_settings = scene.render
    render_settings.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.refresh_devices()
    for device in preferences.devices:
        if device.type in ('CPU','OPTIX'):
            device.use = device.type == 'OPTIX'
    device_names = [device.name for device in preferences.devices if device.type == 'OPTIX' and device.use]
    if not device_names:
        raise RuntimeError('No OptiX GPU is available.')
    scene.cycles.samples = 16
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = .05
    scene.cycles.use_denoising = True
    scene.cycles.seed = base_seed
    scene.cycles.use_animated_seed = False
    render_settings.use_persistent_data = True
    render_settings.resolution_x = render_settings.resolution_y = SIDE
    render_settings.resolution_percentage = 100
    render_settings.pixel_aspect_x = render_settings.pixel_aspect_y = 1.
    render_settings.use_border = False
    render_settings.use_multiview = False
    render_settings.image_settings.file_format = 'PNG'
    render_settings.image_settings.color_mode = 'RGB'
    render_settings.image_settings.color_depth = '8'
    render_settings.image_settings.color_management = 'FOLLOW_SCENE'
    render_settings.use_file_extension = True

    source = bpy.data.texts['generate_tray_anomalies.py']
    scope = {'__name__':'prototype_anomalies'}
    exec(compile(source.as_string(),source.name,'exec'),scope)
    generate = scope['generate_anomalies']
    controls = scene.objects[scope['CONTROL']]
    counts = dict.fromkeys(scope['COUNT_KEYS'], 0)
    generate(seed=base_seed, counts=counts, scene=scene)
    scene.camera = camera
    controls['Enabled'] = controls['Blister amount'] = 1.
    save_mask = normal_and_mask_nodes(scene, temporary)
    save_mask.mute = True
    reference = normal / '0001.png'
    render(scene, reference)
    save_mask.mute = False
    for index in range(2, count+1):
        shutil.copyfile(reference, normal / f'{index:04}.png')

    with (folder/'samples.jsonl').open('x', encoding='utf-8', buffering=1) as catalog:
        for index in range(1, count+1):
            seed = base_seed+index
            variation = random.Random(seed ^ 0x719583)
            counts['crossply_blisters'] = variation.randint(2,8)
            counts['flange_blisters'] = variation.randint(1,4)
            for prefix,key in (('Cross-ply','crossply_blisters'),('Flange','flange_blisters')):
                controls[f'{prefix} blister width scale'] = 1.
                controls[f'{prefix} blister lift scale'] = variation.uniform(.82,1.26)
                for number in range(1,counts[key]+1):
                    controls[f'{prefix} blister {number:02d} width'] = variation.uniform(.72,1.24)
            defects = generate(seed=seed,counts=counts,scene=scene)
            scene.camera = camera
            name = f'{index:04}'
            save_mask.file_name = name
            render(scene, defective/f'{name}.png')
            paths = list(temporary.glob('*.png'))
            if len(paths)!=1:
                raise RuntimeError(f'Expected one mask from render {name}, found {paths}')
            pixels = positive_pixels(paths[0])
            paths[0].replace(masks/f'{name}.png')
            catalog.write(json.dumps({
                'index':index,'anomaly_seed':seed,'normal':f'normal/{name}.png',
                'bubbles':f'bubbles/{name}.png','mask':f'masks/{name}.png','positive_pixels':pixels,
                'cross_ply_count':counts['crossply_blisters'],'flange_count':counts['flange_blisters'],
                'instances':defects
            },ensure_ascii=False)+'\n')
            print(f'PROTOTYPE {index}/{count}, seed {seed}, {pixels} mask pixels',flush=True)
    temporary.rmdir()
    (folder/'dataset.json').write_text(json.dumps({
        'pairs':count,'resolution':[SIDE,SIDE],'format':'8-bit PNG','camera':CAMERA,'device':device_names,
        'samples':16,'noise_threshold':.05,'denoising':True,'elapsed_seconds':round(time.perf_counter()-start,2),
        'normal_images':'Copies of one clean render because the camera, lighting, materials and sampling seed are fixed.',
        'mask':'8-bit grayscale; 255 denotes a visible blister pixel and 0 denotes background.',
        'sample_metadata':'samples.jsonl',
    },indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=BASE_SEED)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    run(args.output.resolve(),args.count,args.seed)
