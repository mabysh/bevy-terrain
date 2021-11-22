mod fly_camera;
mod terrain;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, PrintDiagnosticsPlugin},
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::camera::PerspectiveProjection,
};
use fly_camera::{FlyCamera, FlyCameraPlugin};
use terrain::{move_terrain, setup_terrain2, HeightMapTexture};

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .init_resource::<MovementState>()
        .add_asset::<HeightMapTexture>()
        .add_startup_system(spawn_gizmo.system())
        .add_startup_system(spawn_scene.system())
        .add_startup_system(setup_terrain2.system())
        .add_plugin(FlyCameraPlugin)
        // ------ diagnostics -------
        .add_plugin(PrintDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_system(PrintDiagnosticsPlugin::print_diagnostics_system.system())
        // ------ diagnostics -------
        .add_system(move_terrain.system())
        // .add_system(movement_camera.system())
        .run();
}

struct Robot {
    movement_speed: f32,
    rotation_speed: f32,
}

impl Default for Robot {
    fn default() -> Self {
        Self {
            movement_speed: 10.0,
            rotation_speed: 25.0,
        }
    }
}

struct MainCamera {
    pitch: f32,
    distance: f32,
    aap: f32,
    look_sense: f32,
    zoom_sense: f32,
    init: bool,
}

impl Default for MainCamera {
    fn default() -> Self {
        Self {
            pitch: 2.0,
            distance: 15.0,
            aap: 0.0,
            look_sense: 60.0,
            zoom_sense: 1.0,
            init: false,
        }
    }
}

#[derive(Default)]
struct MovementState {
    mouse_motion_reader: EventReader<MouseMotion>,
    mouse_wheel_reader: EventReader<MouseWheel>,
}

fn spawn_scene(commands: &mut Commands, asset_server: Res<AssetServer>) {
    let camera_entity = commands
        .spawn(Camera3dBundle {
            perspective_projection: PerspectiveProjection {
                far: 100000.0,
                fov: 60f32.to_radians(),
                ..Default::default()
            },
            transform: Transform::from_translation(Vec3::new(0.0, 10.0, 30.0)),
            ..Default::default()
        })
        // .with(MainCamera::default())
        .with(FlyCamera::default())
        .current_entity();

    // let robot_entity = commands
    //     .spawn((
    //         Transform::from_translation(Vec3::zero())
    //             .looking_at(invert_vec(Vec3::new(3.0, 0.0, 3.0)), Vec3::unit_y()),
    //         GlobalTransform::default(),
    //     ))
    //     .with(Robot::default())
    //     .with_children(|parent| {
    //         parent.spawn_scene(asset_server.load("models/robot.gltf"));
    //     })
    //     .current_entity();

    // commands.push_children(robot_entity.unwrap(), &[camera_entity.unwrap()]);

    commands.spawn(LightBundle {
        transform: Transform::from_translation(Vec3::new(4.0, 15.0, 4.0)),
        ..Default::default()
    });
}

fn movement_camera(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mouse_button_input: Res<Input<MouseButton>>,
    mouse_motion_events: Res<Events<MouseMotion>>,
    mouse_wheel_events: Res<Events<MouseWheel>>,
    mut state: ResMut<MovementState>,
    mut robot_query: Query<(&mut Transform, &Robot)>,
    mut camera_query: Query<(&mut Transform, &mut MainCamera)>,
) {
    if let Some((mut robot_transform, robot)) = robot_query.iter_mut().next() {
        if let Some((mut camera_transform, mut main_camera)) = camera_query.iter_mut().next() {
            let old_pitch = main_camera.pitch;
            let old_app = main_camera.aap;
            let old_distance = main_camera.distance;

            // read inputs
            if mouse_button_input.pressed(MouseButton::Left)
                || mouse_button_input.pressed(MouseButton::Right)
            {
                let mut pitch_delta = 0f32;
                let mut aap_delta = 0f32;
                for event in state.mouse_motion_reader.iter(&mouse_motion_events) {
                    pitch_delta += event.delta.y;
                    aap_delta += event.delta.x;
                }
                main_camera.pitch = (main_camera.pitch
                    + (pitch_delta * time.delta_seconds() * main_camera.look_sense))
                    .max(1f32)
                    .min(89f32);
                main_camera.aap =
                    //TODO: divide by 360
                    main_camera.aap + aap_delta * time.delta_seconds() * main_camera.look_sense;
            }
            let mut distance_delta = 0f32;
            for event in state.mouse_wheel_reader.iter(&mouse_wheel_events) {
                distance_delta -= event.y;
            }
            main_camera.distance = (main_camera.distance + distance_delta * main_camera.zoom_sense)
                .max(5.)
                .min(30.);
            let follow_aap = mouse_button_input.pressed(MouseButton::Right);

            let mut mov = Vec2::zero();
            if keyboard_input.pressed(KeyCode::W)
                || (mouse_button_input.pressed(MouseButton::Left)
                    && mouse_button_input.pressed(MouseButton::Right))
            {
                mov.y += 1.;
            }
            if keyboard_input.pressed(KeyCode::S) {
                mov.y -= 1.;
            }
            if keyboard_input.pressed(KeyCode::D) {
                mov.x += 1.;
            }
            if keyboard_input.pressed(KeyCode::A) {
                mov.x -= 1.;
            }

            // move robot
            if mov != Vec2::zero() {
                mov *= time.delta_seconds() * robot.movement_speed;
                if follow_aap {
                    robot_transform.rotate(Quat::from_rotation_y(-main_camera.aap.to_radians()));
                    main_camera.aap = 0.;
                }

                let forward = robot_transform.forward();
                let right = forward.cross(Vec3::unit_y());

                let forward: Vec3 = forward * mov.y;
                let right: Vec3 = right * mov.x;

                robot_transform.translation += Vec3::from(forward + right);
            }

            // move camera
            if main_camera.pitch != old_pitch
                || main_camera.aap != old_app
                || main_camera.distance != old_distance
                || !main_camera.init
            {
                main_camera.init = true;
                let pitch_rad = main_camera.pitch.to_radians();
                let aap_rad = main_camera.aap.to_radians();
                let y = pitch_rad.sin() * main_camera.distance;
                let xz = pitch_rad.cos() * main_camera.distance;
                let x = aap_rad.sin() * xz;
                let z = aap_rad.cos() * xz;

                camera_transform.translation = Vec3::new(x, y, -z);
                let look =
                    Mat4::face_toward(camera_transform.translation, Vec3::unit_y(), Vec3::unit_y());
                camera_transform.rotation = look.to_scale_rotation_translation().1;
            }
        }
    }
}

fn spawn_gizmo(
    commands: &mut Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(0.5, 20.0, 0.5))),
            material: materials.add(Color::rgb(0.0, 1.0, 0.0).into()),
            transform: Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
            ..Default::default()
        })
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(20.0, 0.5, 0.5))),
            material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
            transform: Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            ..Default::default()
        })
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(0.5, 0.5, 20.0))),
            material: materials.add(Color::rgb(0.0, 0.0, 1.0).into()),
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 10.0)),
            ..Default::default()
        });
}

fn invert_vec(to_fix: Vec3) -> Vec3 {
    return -to_fix;
}
