// The entry points. Appended after the effect's own WGSL, which must define:
//
//     fn effect(input: EffectInput) -> vec4<f32>
//
// returning straight-alpha, sRGB-encoded colour. Premultiplication happens
// here, once, because that is where hand-written effects go wrong.

struct EffectRaster {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) instance_id: u32,
    // Positive on all four components means inside the content mask. GPUI has
    // no clip_distance builtin through naga, so this rides as a varying and the
    // fragment shader discards, exactly as the stock quad shaders do.
    @location(1) clip_distances: vec4<f32>,
    @location(2) local: vec2<f32>,
}

@vertex
fn vs_effect(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> EffectRaster {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let instance = effect_instances[instance_id];
    let local = unit_vertex * instance.bounds_size;
    let position = local + instance.bounds_origin;

    var out: EffectRaster;
    out.position = vec4<f32>(
        position / effect_globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0),
        0.0,
        1.0,
    );
    out.instance_id = instance_id;
    out.local = local;
    out.clip_distances = vec4<f32>(
        position.x - instance.clip_origin.x,
        instance.clip_origin.x + instance.clip_size.x - position.x,
        position.y - instance.clip_origin.y,
        instance.clip_origin.y + instance.clip_size.y - position.y,
    );
    return out;
}

/// Which corner radius applies at this point, matching the stock quad shaders
/// so an effect and a quad round identically.
fn effect_corner_radius(center_to_point: vec2<f32>, radii: vec4<f32>) -> f32 {
    if center_to_point.x < 0.0 {
        if center_to_point.y < 0.0 { return radii.x; }
        return radii.w;
    }
    if center_to_point.y < 0.0 { return radii.y; }
    return radii.z;
}

/// Signed distance to the rounded rectangle: negative inside.
fn effect_quad_sdf(local: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let half_size = size / 2.0;
    let center_to_point = local - half_size;
    let corner_radius = effect_corner_radius(center_to_point, radii);
    let corner_center_to_point = abs(center_to_point) - half_size + corner_radius;
    if corner_radius == 0.0 {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    }
    return length(max(vec2<f32>(0.0), corner_center_to_point))
        + min(0.0, max(corner_center_to_point.x, corner_center_to_point.y))
        - corner_radius;
}

@fragment
fn fs_effect(raster: EffectRaster) -> @location(0) vec4<f32> {
    if any(raster.clip_distances < vec4<f32>(0.0)) {
        discard;
    }
    let instance = effect_instances[raster.instance_id];

    var input: EffectInput;
    input.uv = raster.local / instance.bounds_size;
    input.position = raster.local;
    input.size = instance.bounds_size;
    input.scale = instance.scale.x;
    input.params0 = instance.params0;
    input.params1 = instance.params1;
    input.params2 = instance.params2;
    input.params3 = instance.params3;

    let color = effect(input);

    // One pixel of anti-aliasing on the rounded edge, the same treatment the
    // stock quad gets. Without it a rounded effect has a staircase edge that a
    // rounded quad beside it does not.
    let distance = effect_quad_sdf(raster.local, instance.bounds_size, instance.corner_radii);
    let coverage = saturate(0.5 - distance);

    let alpha = color.a * coverage * instance.scale.y;
    let multiplier = select(1.0, alpha, effect_globals.premultiplied_alpha != 0u);
    return vec4<f32>(color.rgb * multiplier, alpha);
}
