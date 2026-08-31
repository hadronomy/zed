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
    // `.x` is device pixels per logical pixel, `.y` is element opacity, and
    // `.zw` are reserved. A bare `f32` here would be a trap: WGSL aligns the
    // `vec3` that would have to follow it to 16 bytes, so the shader and the
    // Rust struct would disagree about the offset of every field after this
    // one, and the effect would read parameters out of the padding.
    scale: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    params4: vec4<f32>,
    params5: vec4<f32>,
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
    params4: vec4<f32>,
    params5: vec4<f32>,
}

/// The floats the application packed, by index.
///
/// Indices run 0..16. Anything higher reads the last float rather than
/// trapping, because a shader is not the place to discover an off-by-one.
fn param(input: EffectInput, index: u32) -> f32 {
    var row = input.params5;
    if index < 4u {
        row = input.params0;
    } else if index < 8u {
        row = input.params1;
    } else if index < 12u {
        row = input.params2;
    } else if index < 16u {
        row = input.params3;
    } else if index < 20u {
        row = input.params4;
    }
    let lane = index & 3u;
    if lane == 0u { return row.x; }
    if lane == 1u { return row.y; }
    if lane == 2u { return row.z; }
    return row.w;
}

struct Hsla {
    h: f32,
    s: f32,
    l: f32,
    a: f32,
}

// Copied from GPUI's own shaders so a colour parameter and a quad's background
// resolve identically. A colour parameter reaches the shader as an Hsla,
// because that is what GPUI's Rust side holds.
fn hsla_to_rgba(hsla: Hsla) -> vec4<f32> {
    let h = hsla.h * 6.0;
    let s = hsla.s;
    let l = hsla.l;
    let a = hsla.a;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(h % 2.0 - 1.0));
    let m = l - c / 2.0;
    var color = vec3<f32>(m);

    if (h >= 0.0 && h < 1.0) {
        color.r += c;
        color.g += x;
    } else if (h >= 1.0 && h < 2.0) {
        color.r += x;
        color.g += c;
    } else if (h >= 2.0 && h < 3.0) {
        color.g += c;
        color.b += x;
    } else if (h >= 3.0 && h < 4.0) {
        color.g += x;
        color.b += c;
    } else if (h >= 4.0 && h < 5.0) {
        color.r += x;
        color.b += c;
    } else {
        color.r += c;
        color.b += x;
    }

    return vec4<f32>(color, a);
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
