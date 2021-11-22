use core::f32;
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
    unimplemented,
};

use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::Indices,
        pipeline::{PipelineDescriptor, PrimitiveTopology, RenderPipeline},
        render_graph::{self, AssetRenderResourcesNode, RenderGraph},
        renderer::RenderResources,
        shader::ShaderStages,
        texture::{Extent3d, TextureDimension, TextureFormat},
    },
    utils::HashMap,
};
use bzip2::bufread::BzDecoder;

pub struct Grid {
    pub step_length: f32,
    pub steps_x: u8,
    pub steps_z: u8,
}

impl From<Grid> for Mesh {
    fn from(grid: Grid) -> Self {
        let verts_x = grid.steps_x as usize + 1;
        let verts_z = grid.steps_z as usize + 1;
        let step_values: Vec<f32> = (0..verts_x.max(verts_z))
            .into_iter()
            .map(|s| s as f32 * grid.step_length)
            .collect();

        // y is calculated in vertex shader
        let mut positions: Vec<[f32; 2]> = Vec::with_capacity(verts_x * verts_z);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(verts_x * verts_z);
        // let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(verts_z * verts_z);

        for z in 0..verts_z {
            for x in 0..verts_x {
                positions.push([step_values[x], step_values[z]]);
                normals.push([0.0, 1.0, 0.0]);
                // uvs.push([
                //     step_values[x] / step_values.last().unwrap(),
                //     step_values[z] / step_values.last().unwrap(),
                // ]);
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.set_indices(Some(Indices::U16(generate_indices(
            grid.steps_x as u16,
            grid.steps_z as u16,
            PrimitiveTopology::LineList,
        ))));
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        // mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh
    }
}

fn generate_indices(steps_x: u16, steps_z: u16, topology: PrimitiveTopology) -> Vec<u16> {
    let verts_by_side = steps_x + 1;
    let cells = steps_x * steps_z;
    let capacity = match topology {
        PrimitiveTopology::LineList => cells as usize * 10,
        PrimitiveTopology::TriangleList => cells as usize * 6,
        _ => unimplemented!(),
    };
    let mut indices: Vec<u16> = Vec::with_capacity(capacity);

    for cell in 0..cells {
        let row = cell / steps_x;
        let bottom_left = cell + row;
        let top_left = bottom_left + verts_by_side;
        let bottom_right = bottom_left + 1;
        let top_right = bottom_right + verts_by_side;
        match topology {
            PrimitiveTopology::LineList => indices.extend_from_slice(&[
                bottom_left,
                bottom_right,
                bottom_right,
                top_right,
                top_right,
                bottom_left,
                top_right,
                top_left,
                top_left,
                bottom_left,
            ]),
            PrimitiveTopology::TriangleList => indices.extend_from_slice(&[
                bottom_left,
                top_right,
                bottom_right,
                top_right,
                bottom_left,
                top_left,
            ]),
            _ => unimplemented!(),
        };
    }
    indices
}

#[derive(PartialEq, Clone, Copy)]
enum LShapePosition {
    BottomRight,
    TopRight,
    TopLeft,
    BottomLeft,
}

#[derive(RenderResources, Default, TypeUuid)]
#[uuid = "93fb26fc-6c05-489b-9029-601edf703b6b"]
pub struct HeightMapTexture {
    texture: Handle<Texture>,
    color: Color,
    scale: f32,
    middle: Vec3,
}

pub struct Terrain {
    levels: u8,
    unit_size: f32,
}

pub struct LShapeVert;
pub struct LShapeHor;

pub struct TerrainLevel {
    level: u8,
    scaled_unit: f32,
    lshape_position: LShapePosition,
    lshape_hor_bottom_right: Transform,
    lshape_hor_top_right: Transform,
    lshape_hor_top_left: Transform,
    lshape_hor_bottom_left: Transform,
    lshape_vert_top_right: Transform,
    lshape_vert_bottom_right: Transform,
    lshape_vert_bottom_left: Transform,
    lshape_vert_top_left: Transform,
    lshape_hor: Entity,
    lshape_vert: Entity,
}

pub fn setup_terrain2(
    commands: &mut Commands,
    asset_server: Res<AssetServer>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_graph: ResMut<RenderGraph>,
    mut heightmap_textures: ResMut<Assets<HeightMapTexture>>,
    mut textures: ResMut<Assets<Texture>>,
) {
    asset_server.watch_for_changes().unwrap();
    // let bytes = include_bytes!("../assets/texture/Terrain_256_bc4.dds");
    let i = image::open(Path::new(
        "assets/texture/ant/ant03_heightmap_2048_rgba.png",
    ))
    .unwrap();
    // let image = image::load_from_memory(bytes).unwrap();
    let pixel = i.as_bytes();
    let tex = Texture::new_fill(
        Extent3d {
            width: 2048,
            height: 2048,
            depth: 1,
        },
        TextureDimension::D2,
        pixel,
        TextureFormat::Rgba8Unorm,
    );
    // Create a new shader pipeline.
    let pipeline_handle = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: asset_server.load::<Shader, _>("shaders/terrain.vert"),
        fragment: Some(asset_server.load::<Shader, _>("shaders/terrain.frag")),
    }));

    render_graph.add_system_node(
        "heightmap_texture",
        AssetRenderResourcesNode::<HeightMapTexture>::new(true),
    );
    render_graph
        .add_node_edge("heightmap_texture", render_graph::base::node::MAIN_PASS)
        .unwrap();

    // let terrain_texture = asset_server.load("texture/Terrain_H_64.png");
    let terrain_texture = textures.add(tex);

    let texture_size = 64u16;
    let unit_size = 1f32;
    let levels = 10;
    let grid_steps = ((texture_size - 4) / 4) as u8;

    let grid_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: grid_steps,
        steps_z: grid_steps,
    }));
    let filler_vert_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: 1,
        steps_z: grid_steps,
    }));
    let filler_hor_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: grid_steps,
        steps_z: 1,
    }));
    let center_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: 1,
        steps_z: 1,
    }));
    let lshape_vert_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: 1,
        steps_z: (texture_size - 2) as u8,
    }));
    let lshape_hor_mesh = meshes.add(Mesh::from(Grid {
        step_length: unit_size,
        steps_x: (texture_size - 3) as u8,
        steps_z: 1,
    }));

    let terrain_entity = commands
        .spawn((
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            GlobalTransform::default(),
        ))
        .with(Terrain { levels, unit_size })
        .current_entity();

    for l in 0..levels {
        let scale = (1 << l) as f32;
        let scaled_unit = scale * unit_size;
        let grid_side = grid_steps as f32 * scaled_unit;
        let level_base = Vec2::new(-grid_side * 2., -grid_side * 2.);

        let grid_tex = heightmap_textures.add(HeightMapTexture {
            texture: terrain_texture.clone(),
            color: Color::YELLOW_GREEN,
            scale: scaled_unit,
            middle: Vec3::zero(),
        });
        let filler_tex = heightmap_textures.add(HeightMapTexture {
            texture: terrain_texture.clone(),
            color: Color::RED,
            scale: scaled_unit,
            middle: Vec3::zero(),
        });
        let lshape_tex = heightmap_textures.add(HeightMapTexture {
            texture: terrain_texture.clone(),
            color: Color::GREEN,
            scale: scaled_unit,
            middle: Vec3::zero(),
        });
        let level_side = (grid_steps * 4 + 1) as f32 * scaled_unit;
        let lshape_hor_bottom_right = Transform {
            translation: Vec3::new(level_base.x, 0.0, level_base.y + level_side),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_hor_top_right = Transform {
            translation: Vec3::new(level_base.x, 0.0, level_base.y - scaled_unit),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_hor_top_left = Transform {
            translation: Vec3::new(level_base.x, 0.0, level_base.y - scaled_unit),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_hor_bottom_left = Transform {
            translation: Vec3::new(level_base.x, 0.0, level_base.y + level_side),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_vert_top_right = Transform {
            translation: Vec3::new(level_base.x + level_side, 0.0, level_base.y - scaled_unit),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_vert_bottom_right = Transform {
            translation: Vec3::new(level_base.x + level_side, 0.0, level_base.y),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_vert_top_left = Transform {
            translation: Vec3::new(level_base.x - scaled_unit, 0.0, level_base.y - scaled_unit),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };
        let lshape_vert_bottom_left = Transform {
            translation: Vec3::new(level_base.x - scaled_unit, 0.0, level_base.y),
            scale: Vec3::new(scale, 1.0, scale),
            ..Default::default()
        };

        let lshape_vert = commands
            .spawn(PbrBundle {
                mesh: lshape_vert_mesh.clone(),
                visible: Visible {
                    is_visible: l != levels - 1,
                    ..Default::default()
                },
                transform: lshape_vert_bottom_right.clone(),
                render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                    pipeline_handle.clone(),
                )]),
                ..Default::default()
            })
            .with(lshape_tex.clone())
            .with(LShapeVert)
            .current_entity();
        let lshape_hor = commands
            .spawn(PbrBundle {
                mesh: lshape_hor_mesh.clone(),
                visible: Visible {
                    is_visible: l != levels - 1,
                    ..Default::default()
                },
                transform: lshape_hor_bottom_left.clone(),
                render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                    pipeline_handle.clone(),
                )]),
                ..Default::default()
            })
            .with(lshape_tex.clone())
            .with(LShapeHor)
            .current_entity();

        let terrain_level = commands
            .spawn((
                Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                GlobalTransform::default(),
            ))
            .with(TerrainLevel {
                scaled_unit,
                level: l,
                lshape_position: LShapePosition::BottomRight,
                lshape_hor_bottom_right: lshape_hor_bottom_right.clone(),
                lshape_hor_top_right: lshape_hor_top_right.clone(),
                lshape_hor_top_left: lshape_hor_top_left.clone(),
                lshape_hor_bottom_left: lshape_hor_bottom_left.clone(),
                lshape_vert_top_right: lshape_vert_top_right.clone(),
                lshape_vert_bottom_right: lshape_vert_bottom_right.clone(),
                lshape_vert_bottom_left: lshape_vert_bottom_left.clone(),
                lshape_vert_top_left: lshape_vert_top_left.clone(),
                lshape_hor: lshape_hor.unwrap(),
                lshape_vert: lshape_vert.unwrap(),
            })
            .current_entity();
        commands.push_children(terrain_entity.unwrap(), &[terrain_level.unwrap()]);
        commands.push_children(
            terrain_level.unwrap(),
            &[lshape_vert.unwrap(), lshape_hor.unwrap()],
        );

        for x in 0..4 {
            for y in 0..4 {
                if inside_ring(x, y) && l > 0 {
                    continue;
                }
                let x_adjustment = if x > 1 { scaled_unit } else { 0. };
                let y_adjustment = if y > 1 { scaled_unit } else { 0. };
                let grid = commands
                    .spawn(PbrBundle {
                        mesh: grid_mesh.clone(),
                        transform: Transform {
                            translation: Vec3::new(
                                level_base.x + (x as f32 * grid_side) + x_adjustment,
                                0.0,
                                level_base.y + (y as f32 * grid_side) + y_adjustment,
                            ),
                            scale: Vec3::new(scale, 1.0, scale),
                            ..Default::default()
                        },
                        render_pipelines: RenderPipelines::from_pipelines(vec![
                            RenderPipeline::new(pipeline_handle.clone()),
                        ]),
                        ..Default::default()
                    })
                    .with(grid_tex.clone())
                    .current_entity();
                commands.push_children(terrain_level.unwrap(), &[grid.unwrap()]);

                if x == 1 {
                    let filler_vert = commands
                        .spawn(PbrBundle {
                            mesh: filler_vert_mesh.clone(),
                            transform: Transform {
                                translation: Vec3::new(
                                    level_base.x + (2. * grid_side),
                                    0.0,
                                    level_base.y + (y as f32 * grid_side) + y_adjustment,
                                ),
                                scale: Vec3::new(scale, 1.0, scale),
                                ..Default::default()
                            },
                            render_pipelines: RenderPipelines::from_pipelines(vec![
                                RenderPipeline::new(pipeline_handle.clone()),
                            ]),
                            ..Default::default()
                        })
                        .with(filler_tex.clone())
                        .current_entity();
                    commands.push_children(terrain_level.unwrap(), &[filler_vert.unwrap()]);
                }
                if y == 1 {
                    let filler_hor = commands
                        .spawn(PbrBundle {
                            mesh: filler_hor_mesh.clone(),
                            transform: Transform {
                                translation: Vec3::new(
                                    level_base.x + (x as f32 * grid_side) + x_adjustment,
                                    0.0,
                                    level_base.y + (2. * grid_side),
                                ),
                                scale: Vec3::new(scale, 1.0, scale),
                                ..Default::default()
                            },
                            render_pipelines: RenderPipelines::from_pipelines(vec![
                                RenderPipeline::new(pipeline_handle.clone()),
                            ]),
                            ..Default::default()
                        })
                        .with(filler_tex.clone())
                        .current_entity();
                    commands.push_children(terrain_level.unwrap(), &[filler_hor.unwrap()]);
                }
            }
        }

        if l == 0 {
            let center = commands
                .spawn(PbrBundle {
                    mesh: center_mesh.clone(),
                    transform: Transform {
                        translation: Vec3::new(
                            level_base.x + grid_side * 2.,
                            0.0,
                            level_base.y + grid_side * 2.,
                        ),
                        ..Default::default()
                    },
                    render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                        pipeline_handle.clone(),
                    )]),
                    ..Default::default()
                })
                .with(filler_tex.clone())
                .current_entity();
            commands.push_children(terrain_level.unwrap(), &[center.unwrap()]);
        }
    }

    fn inside_ring(x: i32, y: i32) -> bool {
        (x == 1 || x == 2) && (y == 1 || y == 2)
    }
}

pub fn move_terrain(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mut heightmap_textures: ResMut<Assets<HeightMapTexture>>,
    tex_q: Query<&Handle<HeightMapTexture>>,
    mut terrain_q: Query<(&mut Transform, &Terrain)>,
    mut level_q: Query<&mut TerrainLevel>,
    mut l_hor_q: Query<(&mut Transform, &LShapeHor)>,
    mut l_vert_q: Query<(&mut Transform, &LShapeVert)>,
) {
    if let Some((mut t_transform, t)) = terrain_q.iter_mut().next() {
        // move terrain
        let mut mov = Vec2::zero();
        if keyboard_input.pressed(KeyCode::Up) {
            mov.y += 1.;
        }
        if keyboard_input.pressed(KeyCode::Down) {
            mov.y -= 1.;
        }
        if keyboard_input.pressed(KeyCode::Right) {
            mov.x += 1.;
        }
        if keyboard_input.pressed(KeyCode::Left) {
            mov.x -= 1.;
        }
        if mov == Vec2::zero() {
            return;
        }
        mov *= time.delta_seconds() * 5.0;
        t_transform.translation.x -= mov.x;
        t_transform.translation.z += mov.y;

        let map: HashMap<u32, (Vec3, LShapePosition)> = (0..t.levels)
            .map(|level| {
                let unit = (1 << level) as f32 * t.unit_size;
                let next_unit = (1 << level + 1) as f32 * t.unit_size;
                // TODO optimize
                let pos = (t_transform.translation / unit).floor() * unit;
                let next_pos = (t_transform.translation / next_unit).floor() * next_unit;
                let x_offset = pos.x - next_pos.x;
                let z_offset = pos.z - next_pos.z;
                let lshape_pos;
                if x_offset == 0. && z_offset == 0. {
                    lshape_pos = LShapePosition::BottomRight;
                } else if x_offset == unit && z_offset == 0. {
                    lshape_pos = LShapePosition::BottomLeft;
                } else if x_offset == unit && z_offset == unit {
                    lshape_pos = LShapePosition::TopLeft;
                } else {
                    lshape_pos = LShapePosition::TopRight;
                }

                (unit as u32, (pos, lshape_pos))
            })
            .collect();

        // TODO use par_iter_mut ?
        for mut level in level_q.iter_mut() {
            if let Some((_, lshape_pos)) = map.get(&(level.scaled_unit as u32)) {
                if level.lshape_position == *lshape_pos {
                    continue;
                }
                if let Ok((mut l_vert_t, _)) = l_vert_q.get_mut(level.lshape_vert) {
                    if let Ok((mut l_hor_t, _)) = l_hor_q.get_mut(level.lshape_hor) {
                        match lshape_pos {
                            LShapePosition::BottomRight => {
                                *l_vert_t = level.lshape_vert_bottom_right;
                                *l_hor_t = level.lshape_hor_bottom_left;
                            }
                            LShapePosition::TopRight => {
                                *l_vert_t = level.lshape_vert_top_right;
                                *l_hor_t = level.lshape_hor_top_left;
                            }
                            LShapePosition::TopLeft => {
                                *l_vert_t = level.lshape_vert_top_left;
                                *l_hor_t = level.lshape_hor_top_right;
                            }
                            LShapePosition::BottomLeft => {
                                *l_vert_t = level.lshape_vert_bottom_left;
                                *l_hor_t = level.lshape_hor_bottom_right;
                            }
                        }
                        level.lshape_position = *lshape_pos;
                    }
                }
            }
        }
        for handle in tex_q.iter() {
            if let Some(t) = heightmap_textures.get_mut(handle) {
                if let Some((new_pos, _)) = map.get(&(t.scale as u32)) {
                    if t.middle != *new_pos {
                        t.middle = *new_pos;
                    }
                }
            }
        }
    }
}

pub fn test(
    time: Res<Time>,
    mut heightmap_textures: ResMut<Assets<HeightMapTexture>>,
    mut textures: ResMut<Assets<Texture>>,
    query: Query<&Handle<HeightMapTexture>>,
) {
    let sin = time.seconds_since_startup().sin() as f32;
    for handle in query.iter() {
        if let Some(t) = heightmap_textures.get_mut(handle) {
            println!("{:?}", t.color);
            t.color = Color::rgb_linear(sin, sin, sin);
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{
        fmt::format,
        fs::File,
        io::{BufReader, Read},
        path::Path,
    };

    use bevy::{render::pipeline::PrimitiveTopology, utils::Instant};
    use bzip2::{read::BzDecoder, write::BzEncoder, Compression};
    use image::{
        bmp::{BmpDecoder, BmpEncoder},
        dds::DdsDecoder,
        dxt::DxtDecoder,
        farbfeld::{FarbfeldDecoder, FarbfeldEncoder},
        imageops,
        png::{PngDecoder, PngEncoder},
        ColorType, GenericImageView, ImageBuffer, ImageDecoder, ImageDecoderExt, ImageEncoder,
        Pixel, Rgba,
    };
    use zerocopy::AsBytes;

    use super::generate_indices;

    #[test]
    fn grid_indices_test() {
        let indices = generate_indices(3, 3, PrimitiveTopology::TriangleList);
        assert_eq!(
            indices,
            vec![
                0, 1, 5, 5, 4, 0, 1, 2, 6, 6, 5, 1, 2, 3, 7, 7, 6, 2, 4, 5, 9, 9, 8, 4, 5, 6, 10,
                10, 9, 5, 6, 7, 11, 11, 10, 6, 8, 9, 13, 13, 12, 8, 9, 10, 14, 14, 13, 9, 10, 11,
                15, 15, 14, 10
            ]
        );
    }

    #[test]
    fn filler_horizontal_indices_test() {
        let indices = generate_indices(3, 1, PrimitiveTopology::TriangleList);
        assert_eq!(
            indices,
            vec![0, 1, 5, 5, 4, 0, 1, 2, 6, 6, 5, 1, 2, 3, 7, 7, 6, 2]
        );
    }

    #[test]
    fn filler_vertical_indices_test() {
        let indices = generate_indices(1, 3, PrimitiveTopology::TriangleList);
        assert_eq!(
            indices,
            vec![0, 1, 3, 3, 2, 0, 2, 3, 5, 5, 4, 2, 4, 5, 7, 7, 6, 4]
        );
    }

    #[test]
    fn image_test() {
        let i = image::open(Path::new("assets/texture/test/small_island_heightmap.png")).unwrap();
        // println!("{:?}", i.as_flat_samples_u16());
        let bytes = i
            .as_bytes()
            .chunks(2)
            .flat_map(|ch| vec![ch[0], ch[1], 255])
            .collect::<Vec<u8>>();
        image::save_buffer(
            "assets/texture/test/small_island_heightmap_rgba.png",
            bytes.as_slice(),
            6144,
            4096,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }

    #[test]
    fn create_rgba() {
        let name = "ant03_heightmap";
        let min_res = 32;
        let max_res = 32;
        let mut curr_res = min_res;
        while curr_res <= max_res {
            let i = image::open(Path::new(
                format!("assets/texture/ant/{}_{}.png", name, curr_res).as_str(),
            ))
            .unwrap();
            let bytes = i
                .as_bytes()
                .chunks(2)
                .flat_map(|ch| vec![ch[0], ch[1], 255, 255])
                .collect::<Vec<u8>>();
            image::save_buffer(
                format!("assets/texture/ant/{}_{}_rgba.png", name, curr_res),
                bytes.as_bytes(),
                curr_res,
                curr_res,
                image::ColorType::Rgba8,
            )
            .unwrap();
            curr_res *= 2;
        }
    }

    #[test]
    fn residuals() {
        let max = 4096;
        let min = 64;
        let mut curr = max;
        while curr >= min {
            let mut i_finer = image::open(Path::new(
                format!("assets/texture/ant/ant03_heightmap_{}_rgba.png", curr).as_str(),
            ))
            .unwrap()
            .to_rgba8();
            let i_coarser = image::open(Path::new(
                format!("assets/texture/ant/ant03_heightmap_{}_rgba.png", curr / 2).as_str(),
            ))
            .unwrap()
            .to_rgba8();
            let width = i_finer.width();
            let height = i_finer.height();
            for x in 0..width {
                for y in 0..height {
                    let f_px = i_finer.get_pixel(x, y).to_rgba();
                    if x % 2 == 0 && y % 2 == 0 {
                        let c_px = i_coarser.get_pixel(x / 2, y / 2).to_rgba();
                        i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], c_px[0], c_px[1]]));
                    } else if x % 2 != 0 && y % 2 == 0 && x < width - 2 {
                        let c_px_prev = i_coarser.get_pixel(x / 2, y / 2).to_rgba();
                        let c_px_next = i_coarser.get_pixel((x + 1) / 2, y / 2).to_rgba();
                        let c_prev = c_px_prev[1] as u32 * 256 + c_px_prev[0] as u32;
                        let c_next = c_px_next[1] as u32 * 256 + c_px_next[0] as u32;
                        let c = (c_prev + c_next) / 2;
                        i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], c as u8, (c >> 8) as u8]));
                    } else if x % 2 == 0 && y % 2 != 0 && y < height - 2 {
                        let c_px_prev = i_coarser.get_pixel(x / 2, y / 2).to_rgba();
                        let c_px_next = i_coarser.get_pixel(x / 2, (y + 1) / 2).to_rgba();
                        let c_prev = c_px_prev[1] as u32 * 256 + c_px_prev[0] as u32;
                        let c_next = c_px_next[1] as u32 * 256 + c_px_next[0] as u32;
                        let c = (c_prev + c_next) / 2;
                        i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], c as u8, (c >> 8) as u8]));
                    } else if x % 2 != 0 && y % 2 != 0 && x < width - 2 && y < height - 2 {
                        let c_px_prev = i_coarser.get_pixel((x + 1) / 2, y / 2).to_rgba();
                        let c_px_next = i_coarser.get_pixel(x / 2, (y + 1) / 2).to_rgba();
                        let c_prev = c_px_prev[1] as u32 * 256 + c_px_prev[0] as u32;
                        let c_next = c_px_next[1] as u32 * 256 + c_px_next[0] as u32;
                        let c = (c_prev + c_next) / 2;
                        i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], c as u8, (c >> 8) as u8]));
                    } else {
                        i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], 0, 0]));
                    }
                    // i_finer.put_pixel(x, y, Rgba([f_px[0], f_px[1], res as u8, (res >> 8) as u8]))
                }
            }
            i_finer
                .save(Path::new(
                    format!("assets/texture/ant/ant03_heightmap_{}_rgba.png", curr).as_str(),
                ))
                .unwrap();
            curr = curr / 2;
        }

        // let max_a = max_res as u8;
        // let max_b = max_res >> 8;
        // println!("desctucted: {}, {}", max_a, max_b);
        // println!("reconstructed: {}", ((max_b as u16) << 8) | max_a as u16);
        // println!("reconstructed: {}", max_b as u16 * 256 + max_a as u16);
    }

    #[test]
    fn decode_bmp() {
        let f = File::open("assets/texture/test/ant01_heightmap_rgba.bmp").unwrap();
        let mut decoder = BmpDecoder::new(f).unwrap();

        let mut buf: Vec<u8> = vec![0; decoder.total_bytes() as usize];
        decoder.read_rect(0, 0, 8, 8, buf.as_bytes_mut()).unwrap();

        let mut f = File::create("assets/texture/test/ant01.bmp").unwrap();
        let encoder = BmpEncoder::new(&mut f);
        encoder
            .write_image(buf.as_bytes(), 64, 64, ColorType::Rgba8)
            .unwrap();
    }

    #[test]
    fn dxt_bytes() {
        let now = Instant::now();
        let d = PngDecoder::new(File::open("assets/texture/Terrain_H_256.png").unwrap()).unwrap();
        // let i = image::open(Path::new("assets/texture/Terrain_H_256.png")).unwrap();
        // let mut buf: Vec<u8> = vec![0; d.total_bytes() as usize];
        // d.read_image(&mut buf).unwrap();

        // println!("{}", now.elapsed().as_millis());
        println!("{}", d.total_bytes());
    }

    #[test]
    fn decode_rect_test() {
        let i = image::open(Path::new("assets/texture/test/terrain-00.bmp")).unwrap();
        i.save(Path::new("assets/texture/test/terrain-00.png"))
            .unwrap()
    }
}
