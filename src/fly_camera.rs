use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    math::clamp,
    prelude::*,
};

pub struct FlyCamera {
    /// The speed the FlyCamera moves at. Defaults to `1.0`
    pub speed: f32,
    /// The sensitivity of the FlyCamera's motion based on mouse movement. Defaults to `3.0`
    pub sensitivity: f32,
    /// The current pitch of the FlyCamera in degrees. This value is always up-to-date, enforced by [FlyCameraPlugin](struct.FlyCameraPlugin.html)
    pub pitch: f32,
    /// The current pitch of the FlyCamera in degrees. This value is always up-to-date, enforced by [FlyCameraPlugin](struct.FlyCameraPlugin.html)
    pub yaw: f32,
    /// Key used to move forward. Defaults to `W`
    pub key_forward: KeyCode,
    /// Key used to move backward. Defaults to `S
    pub key_backward: KeyCode,
    /// Key used to move left. Defaults to `A`
    pub key_left: KeyCode,
    /// Key used to move right. Defaults to `D`
    pub key_right: KeyCode,
    /// Key used to move up. Defaults to `Space`
    pub key_up: KeyCode,
    /// Key used to move forward. Defaults to `LShift`
    pub key_down: KeyCode,
    /// If `false`, disable keyboard control of the camera. Defaults to `true`
    pub enabled: bool,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            speed: 1.5,
            sensitivity: 12.0,
            pitch: 0.0,
            yaw: 0.0,
            key_forward: KeyCode::W,
            key_backward: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::Space,
            key_down: KeyCode::LShift,
            enabled: true,
        }
    }
}

fn camera_movement_system(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mut query: Query<(&FlyCamera, &mut Transform)>,
) {
    for (options, mut transform) in query.iter_mut() {
        let mut mov = Vec3::zero();
        if keyboard_input.pressed(options.key_forward) {
            mov.z += 1.;
        }
        if keyboard_input.pressed(options.key_backward) {
            mov.z -= 1.;
        }
        if keyboard_input.pressed(options.key_right) {
            mov.x += 1.;
        }
        if keyboard_input.pressed(options.key_left) {
            mov.x -= 1.;
        }
        if keyboard_input.pressed(options.key_up) {
            mov.y += 1.;
        }
        if keyboard_input.pressed(options.key_down) {
            mov.y -= 1.;
        }

        if mov != Vec3::zero() {
            mov *= time.delta_seconds() * options.speed;
            // TODO remove minus when forward is fixed
            let forward = -transform.forward();
            let right = forward.cross(Vec3::unit_y());

            let forward: Vec3 = forward * mov.z;
            let right: Vec3 = right * mov.x;
            let up = Vec3::unit_y() * mov.y;

            transform.translation += Vec3::from(forward + right + up);
        }
    }
}

#[derive(Default)]
struct State {
    mouse_motion_event_reader: EventReader<MouseMotion>,
    mouse_wheel_reader: EventReader<MouseWheel>,
}

fn mouse_motion_system(
    time: Res<Time>,
    mut state: ResMut<State>,
    mouse_button_input: Res<Input<MouseButton>>,
    mouse_motion_events: Res<Events<MouseMotion>>,
    mouse_wheel_events: Res<Events<MouseWheel>>,
    mut query: Query<(&mut FlyCamera, &mut Transform)>,
) {
    let mut delta: Vec2 = Vec2::zero();
    if mouse_button_input.pressed(MouseButton::Left)
        || mouse_button_input.pressed(MouseButton::Right)
    {
        for event in state.mouse_motion_event_reader.iter(&mouse_motion_events) {
            delta += event.delta;
        }
    }
    let mut speed_delta = 0.0;
    for event in state.mouse_wheel_reader.iter(&mouse_wheel_events) {
        speed_delta += event.y;
    }
    if (delta == Vec2::zero() || delta.is_nan()) && speed_delta == 0.0 {
        return;
    }

    for (mut options, mut transform) in query.iter_mut() {
        if !options.enabled {
            continue;
        }
        speed_delta *= time.delta_seconds() * options.speed * 5.0;
        options.speed = (options.speed + speed_delta).min(200.0).max(0.0);
        options.yaw -= delta.x * options.sensitivity * time.delta_seconds();
        options.pitch += delta.y * options.sensitivity * time.delta_seconds();

        options.pitch = clamp(options.pitch, -89.9, 89.9);
        // println!("pitch: {}, yaw: {}", options.pitch, options.yaw);

        let yaw_radians = options.yaw.to_radians();
        let pitch_radians = options.pitch.to_radians();

        transform.rotation = Quat::from_axis_angle(Vec3::unit_y(), yaw_radians)
            * Quat::from_axis_angle(-Vec3::unit_x(), pitch_radians);
    }
}

/**
Include this plugin to add the systems for the FlyCamera bundle.

```no_run
fn main() {
    App::build().add_plugin(FlyCameraPlugin);
}
```

**/

pub struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.init_resource::<State>()
            .add_system(camera_movement_system.system())
            .add_system(mouse_motion_system.system());
    }
}
