// The effect ABI. Every effect module is this text, then the effect's own WGSL,
// then `epilogue.wgsl`. Changing anything here recompiles every effect in every
// application, so treat it as a published interface.

struct EffectGlobals {
    viewport_size: vec2<f32>,
    // 1 when the surface wants premultiplied colour. Negotiated per platform,
    // so an effect must never assume one or the other.
    premultiplied_alpha: u32,
    _pad: u32,
}

struct EffectInstance {
    bounds_origin: vec2<f32>,
    bounds_size: vec2<f32>,
    clip_origin: vec2<f32>,
    clip_size: vec2<f32>,
    corner_radii: vec4<f32>,
    // `.x` is device pixels per logical pixel; the other three lanes are
    // reserved. A bare `f32` here would be a trap: WGSL aligns the `vec3` that
    // would have to follow it to 16 bytes, so the shader and the Rust struct
    // would disagree about the offset of every field after this one, and the
    // effect would read parameters out of the padding.
    scale: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
}

@group(0) @binding(0) var<uniform> effect_globals: EffectGlobals;
@group(0) @binding(1) var<storage, read> effect_instances: array<EffectInstance>;

/// Everything an effect may read.
struct EffectInput {
    /// 0..1 across the element.
    uv: vec2<f32>,
    /// Device pixels from the element's top-left corner.
    position: vec2<f32>,
    /// The element's size in device pixels.
    size: vec2<f32>,
    /// Device pixels per logical pixel. A radius in logical pixels is
    /// `radius * input.scale` here.
    scale: f32,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
}

/// The sixteen floats the application packed, by index.
///
/// Indices run 0..16. Anything higher reads the last float rather than
/// trapping, because a shader is not the place to discover an off-by-one.
fn param(input: EffectInput, index: u32) -> f32 {
    var row = input.params3;
    if index < 4u {
        row = input.params0;
    } else if index < 8u {
        row = input.params1;
    } else if index < 12u {
        row = input.params2;
    }
    let lane = index & 3u;
    if lane == 0u { return row.x; }
    if lane == 1u { return row.y; }
    if lane == 2u { return row.z; }
    return row.w;
}

/// Four consecutive floats as a colour.
fn param_rgba(input: EffectInput, index: u32) -> vec4<f32> {
    return vec4<f32>(
        param(input, index),
        param(input, index + 1u),
        param(input, index + 2u),
        param(input, index + 3u),
    );
}

// The render targets are BGRA8 UNORM on every backend we drive, so GPUI blends
// values that are still sRGB-encoded. Mixing encoded values darkens the result.
// Convert, mix, convert back — these two are here so no effect has to remember.

fn to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let cutoff = step(encoded, vec3<f32>(0.04045));
    let low = encoded / 12.92;
    let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));
    return mix(high, low, cutoff);
}

fn to_encoded(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = step(linear, vec3<f32>(0.0031308));
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055;
    return mix(high, low, cutoff);
}
