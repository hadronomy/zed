//! GPU effects, authored once in WGSL and translated for each backend.
//!
//! The three renderers speak three shading languages, so an effect cannot be
//! written once in the language a backend compiles. It can be written once in
//! WGSL: naga translates to MSL and to HLSL, and Blade takes WGSL directly.
//!
//! This module owns that translation and nothing else. Pipelines, textures and
//! device handles stay in the renderers, because only a renderer knows what a
//! pipeline is on its platform.
//!
//! An effect is a struct and a fragment function. Derive [`Effect`] and the
//! accessors the shader calls are generated from the field names:
//!
//! ```ignore
//! #[derive(Effect)]
//! #[effect(name = "grain", source = "grain.wgsl")]
//! struct Grain {
//!     amount: f32,
//! }
//! ```
//!
//! ```wgsl
//! fn effect(input: EffectInput) -> vec4<f32> {
//!     return vec4<f32>(input.uv, 0.0, amount(input));
//! }
//! ```
//!
//! [`PREAMBLE`] declares what that function may read and [`EPILOGUE`] supplies
//! the entry points around it. Together they are the ABI: changing either
//! recompiles every effect in every application.
//!
//! Effects live in application crates. This module is the mechanism.

use std::sync::{OnceLock, RwLock};

// A trait and a derive of the same name, from one path, as `serde::Serialize`
// does. `crate::Effect` is already taken by the app's deferred-work enum.
pub use gpui_macros::Effect;

use anyhow::{Context as _, Result, anyhow, bail};

/// Declarations an effect may use. Prepended to every effect module.
pub const PREAMBLE: &str = include_str!("effect/preamble.wgsl");

/// Entry points wrapping the effect's own function. Appended to every module.
pub const EPILOGUE: &str = include_str!("effect/epilogue.wgsl");

/// Vertex entry point that [`EPILOGUE`] defines.
pub const VERTEX_ENTRY: &str = "vs_effect";

/// Fragment entry point that [`EPILOGUE`] defines.
pub const FRAGMENT_ENTRY: &str = "fs_effect";

/// How many floats an effect can hand its shader.
///
/// Six rows of four. An effect that wants more than this is telling us it should
/// be two effects.
pub const PARAM_COUNT: usize = 24;

/// A registered effect.
///
/// Cheap to copy and stable for the life of the process, so a renderer can hold
/// one across frames without borrowing the registry.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct EffectId(pub u32);

/// An effect's name and source, as the application registered it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EffectDef {
    /// Identifies the effect in errors, in GPU debug labels, and in the
    /// registry. Must be unique and stable.
    pub name: &'static str,
    /// The effect's own WGSL. Must define
    /// `fn effect(input: EffectInput) -> vec4<f32>`.
    pub wgsl: &'static str,
    /// The values the shader reads, in the order the application writes them.
    /// GPUI generates an accessor for each, so neither side spells a slot.
    pub parameters: &'static [Parameter],
}

/// One value an effect hands its shader, and the accessor generated for it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Parameter {
    /// The name of the generated accessor function.
    pub name: &'static str,
    /// What that accessor returns, and how many slots it occupies.
    pub kind: ParameterKind,
}

/// What a [`Parameter`]'s generated accessor returns.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParameterKind {
    /// One float; the accessor returns `f32`.
    Scalar,
    /// Four floats holding an [`crate::Hsla`]; the accessor returns
    /// straight-alpha `vec4<f32>`, converted by the same code that resolves a
    /// quad's background.
    Color,
}

impl ParameterKind {
    /// How many of the sixteen slots this occupies.
    pub const fn slots(self) -> usize {
        match self {
            ParameterKind::Scalar => 1,
            ParameterKind::Color => 4,
        }
    }
}

/// Something an application paints through a shader.
///
/// Derive this rather than implementing it. The derive takes the accessor names
/// from the field names and the slot order from the field order, so the struct
/// is the only place either is decided:
///
/// ```ignore
/// #[derive(Effect)]
/// #[effect(name = "grain", source = "grain.wgsl")]
/// struct Grain {
///     amount: f32,
///     size: f32,
/// }
/// ```
///
/// The shader then reads `amount(input)` and `size(input)`. Renaming a field
/// renames the accessor, so a shader left behind fails to translate instead of
/// reading the wrong slot.
pub trait Effect: 'static {
    /// Identifies the effect in the registry and in errors. Unique and stable.
    const NAME: &'static str;
    /// WGSL defining `fn effect(input: EffectInput) -> vec4<f32>`.
    const SOURCE: &'static str;
    /// The values the shader reads, in the order [`Effect::params`] writes.
    const PARAMETERS: &'static [Parameter];

    /// The floats the shader reads, in [`Effect::PARAMETERS`] order.
    fn params(&self) -> [f32; PARAM_COUNT];

    /// The registry entry for this effect.
    fn definition() -> EffectDef {
        EffectDef {
            name: Self::NAME,
            wgsl: Self::SOURCE,
            parameters: Self::PARAMETERS,
        }
    }
}

/// The shading language a backend compiles.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ShaderTarget {
    /// Blade, which compiles WGSL itself.
    Wgsl,
    /// Metal Shading Language.
    Msl,
    /// High Level Shading Language at Shader Model 5.0, which is what Direct3D
    /// 11 accepts. This is the floor: an effect that needs wave intrinsics
    /// compiles on the other two backends and fails here.
    Hlsl,
}

/// Resource slots the generated module uses, and which each renderer must bind.
///
/// These are one set of numbers shared by three backends. If a renderer binds a
/// different slot than the translation targets, the effect draws garbage rather
/// than failing, so the two are stated once, here, and referenced from both
/// sides.
pub mod slots {
    /// Bind group holding the globals and the instance buffer.
    pub const GROUP: u32 = 0;
    /// `EffectGlobals`, a uniform buffer.
    pub const GLOBALS: u32 = 0;
    /// `array<EffectInstance>`, a read-only storage buffer.
    pub const INSTANCES: u32 = 1;

    /// Metal buffer index for the globals.
    pub const MSL_GLOBALS_BUFFER: u8 = 0;
    /// Metal buffer index for the instances.
    pub const MSL_INSTANCES_BUFFER: u8 = 1;

    /// HLSL `cbuffer` register for the globals: `register(b0)`.
    pub const HLSL_GLOBALS_REGISTER: u32 = 0;
    /// HLSL `StructuredBuffer` register for the instances: `register(t1)`.
    /// Slot 1, because `t0` holds the source texture.
    pub const HLSL_INSTANCES_REGISTER: u32 = 1;

    /// Bind group holding what the effect is applied to.
    pub const SOURCE_GROUP: u32 = 1;
    /// The captured content.
    ///
    /// Sampled filtered, and transparent outside its bounds. Both halves of
    /// that are load-bearing for anything that reads more than one texel.
    ///
    /// Filtering is not a quality nicety here: a wide kernel is affordable
    /// precisely because one filtered tap can stand for two texels. And a
    /// capture is the element's bounds and nothing more, so past its edge there
    /// is genuinely no content — repeating would wrap the far side of a card
    /// into the near one, and clamping would smear the edge texel out into a
    /// streak. Reading transparent is the only answer that says what is true,
    /// and it lets a kernel run off the edge and fade out on its own instead of
    /// making every effect bounds-check its taps.
    ///
    /// Metal takes this as a `constexpr sampler` written into the generated
    /// shader, which is how GPUI's own Metal shaders declare theirs. Direct3D
    /// has no inline sampler, so a renderer-side one sits at
    /// [`HLSL_SOURCE_SAMPLER_REGISTER`] and has to agree.
    pub const SOURCE_TEXTURE: u32 = 0;

    /// Metal texture index for the source.
    pub const MSL_SOURCE_TEXTURE: u8 = 0;

    /// HLSL register for the source texture: `register(t0)`.
    pub const HLSL_SOURCE_REGISTER: u32 = 0;
    /// HLSL register for its sampler: `register(s0)`. Metal has no equivalent,
    /// because [`super::SAMPLING`] is written into the Metal shader itself.
    pub const HLSL_SOURCE_SAMPLER_REGISTER: u32 = 0;

    /// The sampler binding within [`SOURCE_GROUP`].
    pub const SOURCE_SAMPLER: u32 = 1;
}


/// Frame-wide data the shader reads. Mirrors `EffectGlobals` in the preamble.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct EffectGlobals {
    /// The drawable size in device pixels.
    pub viewport_size: [f32; 2],
    /// Non-zero when the surface wants premultiplied colour. Negotiated per
    /// platform, so the epilogue asks rather than assuming.
    pub premultiplied_alpha: u32,
    _pad: u32,
}

impl EffectGlobals {
    /// Build the globals for one frame.
    pub fn new(viewport_size: [f32; 2], premultiplied_alpha: bool) -> Self {
        Self {
            viewport_size,
            premultiplied_alpha: premultiplied_alpha as u32,
            _pad: 0,
        }
    }
}

/// Per-instance data the shader reads. Mirrors `EffectInstance` in the preamble.
///
/// `#[repr(C)]` with no implicit padding: every field is 8- or 16-byte aligned
/// by construction, because this is memcpy'd into a GPU buffer and a padding
/// byte read as data is undefined behaviour on the CPU side and a wrong pixel on
/// the GPU side.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct EffectInstance {
    /// Top-left of the element, in device pixels.
    pub bounds_origin: [f32; 2],
    /// Element size, in device pixels.
    pub bounds_size: [f32; 2],
    /// Content mask origin, in device pixels.
    pub clip_origin: [f32; 2],
    /// Content mask size, in device pixels.
    pub clip_size: [f32; 2],
    /// Top-left, top-right, bottom-right, bottom-left, in device pixels.
    pub corner_radii: [f32; 4],
    /// Device pixels per logical pixel.
    pub scale: f32,
    /// The element opacity in force when the effect was painted. Applied by the
    /// epilogue rather than by the effect, so an effect cannot forget it or
    /// apply it twice.
    pub opacity: f32,
    _pad: [f32; 2],
    /// The application's floats.
    pub params: [f32; PARAM_COUNT],
}

impl EffectInstance {
    /// Build an instance. The padding stays private so it cannot be filled with
    /// something a future field would collide with.
    pub fn new(
        bounds_origin: [f32; 2],
        bounds_size: [f32; 2],
        clip_origin: [f32; 2],
        clip_size: [f32; 2],
        corner_radii: [f32; 4],
        scale: f32,
        opacity: f32,
        params: [f32; PARAM_COUNT],
    ) -> Self {
        Self {
            bounds_origin,
            bounds_size,
            clip_origin,
            clip_size,
            corner_radii,
            scale,
            opacity,
            _pad: [0.0; 2],
            params,
        }
    }
}

static REGISTRY: OnceLock<RwLock<Vec<EffectDef>>> = OnceLock::new();

fn registry() -> &'static RwLock<Vec<EffectDef>> {
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register an effect and get a handle to it.
///
/// Registering the same name twice returns the first handle rather than adding a
/// second entry, so a test that registers an application's whole catalogue can
/// run beside an application that already did.
///
/// Registration does not compile anything. Use [`validate_all`] in a test to
/// find a broken shader before a frame does.
pub fn register(def: EffectDef) -> EffectId {
    // Element code calls this on every paint to turn a type into a handle, so
    // the already-registered path takes a read lock and the write lock is only
    // reached once per effect in the life of the process.
    if let Some(id) = lookup(def.name) {
        return id;
    }
    let mut effects = registry().write().unwrap();
    if let Some(index) = effects.iter().position(|existing| existing.name == def.name) {
        return EffectId(index as u32);
    }
    effects.push(def);
    EffectId((effects.len() - 1) as u32)
}

/// The handle for an effect that is already registered.
pub fn lookup(name: &str) -> Option<EffectId> {
    registry()
        .read()
        .unwrap()
        .iter()
        .position(|existing| existing.name == name)
        .map(|index| EffectId(index as u32))
}

/// The definition behind a handle.
pub fn definition(id: EffectId) -> Option<EffectDef> {
    registry().read().unwrap().get(id.0 as usize).copied()
}

/// The size the shader believes [`EffectInstance`] is, in bytes.
///
/// Asks naga rather than trusting a comment. WGSL and Rust apply different
/// alignment rules — `vec3<f32>` aligns to 16 in one and 4 in the other — so the
/// two structs can drift apart without either side failing to compile. The
/// symptom is an effect reading its parameters out of padding, which looks like
/// a broken shader rather than a broken struct.
pub fn shader_instance_size() -> Result<u32> {
    let module = naga::front::wgsl::parse_str(PREAMBLE)
        .map_err(|error| anyhow!("the preamble is not valid WGSL:\n{}", error.emit_to_string(PREAMBLE)))?;
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .context("laying out the preamble's types")?;
    for (handle, ty) in module.types.iter() {
        if ty.name.as_deref() == Some("EffectInstance") {
            return Ok(layouter[handle].size);
        }
    }
    bail!("the preamble declares no `EffectInstance`")
}

/// Every registered effect, in registration order.
pub fn registered() -> Vec<(EffectId, EffectDef)> {
    registry()
        .read()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, def)| (EffectId(index as u32), *def))
        .collect()
}

/// The complete WGSL module for an effect: preamble, the effect, entry points.
pub fn module_source(def: &EffectDef) -> String {
    format!(
        "{PREAMBLE}\n{accessors}\n// ---- {name} ----\n{wgsl}\n// ---- entry points ----\n{EPILOGUE}",
        accessors = accessors(def),
        name = def.name,
        wgsl = def.wgsl,
    )
}

/// One WGSL accessor per declared parameter, so the shader names its inputs.
///
/// Generating the shader-side declaration from the Rust side is what GPUI
/// already does for its own primitives, where `build.rs` runs cbindgen over the
/// scene types into `scene.h`. This is the same trade at the level of an
/// application's parameters.
fn accessors(def: &EffectDef) -> String {
    let mut source = String::new();
    let mut slot = 0;
    for parameter in def.parameters {
        let name = parameter.name;
        match parameter.kind {
            ParameterKind::Scalar => source.push_str(&format!(
                "fn {name}(input: EffectInput) -> f32 {{ return param(input, {slot}u); }}\n"
            )),
            ParameterKind::Color => source.push_str(&format!(
                "fn {name}(input: EffectInput) -> vec4<f32> {{ return hsla_to_rgba(Hsla(\
                 param(input, {slot}u), param(input, {h}u), param(input, {l}u), \
                 param(input, {a}u))); }}\n",
                h = slot + 1,
                l = slot + 2,
                a = slot + 3,
            )),
        }
        slot += parameter.kind.slots();
    }
    source
}

/// How many of the sixteen slots a definition's parameters occupy.
fn slots_used(def: &EffectDef) -> usize {
    def.parameters
        .iter()
        .map(|parameter| parameter.kind.slots())
        .sum()
}

/// Translate an effect into what a backend compiles.
///
/// Errors carry the effect's name and naga's message. A shader error that only
/// says "line 214" is useless, because line 214 is in the preamble for every
/// effect in the application.
pub fn translate(def: &EffectDef, target: ShaderTarget) -> Result<String> {
    let slots = slots_used(def);
    if slots > PARAM_COUNT {
        bail!(
            "effect `{}` needs {slots} parameter slots, but the limit is {PARAM_COUNT}",
            def.name
        );
    }
    let source = module_source(def);
    let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
        anyhow!(
            "effect `{}` is not valid WGSL:\n{}",
            def.name,
            error.emit_to_string(&source)
        )
    })?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| {
        anyhow!(
            "effect `{}` did not validate:\n{}",
            def.name,
            error.emit_to_string(&source)
        )
    })?;

    match target {
        ShaderTarget::Wgsl => {
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .with_context(|| format!("writing WGSL for effect `{}`", def.name))
        }
        ShaderTarget::Msl => write_msl(def, &module, &info),
        ShaderTarget::Hlsl => write_hlsl(def, &module, &info),
    }
}

fn resource(group: u32, binding: u32) -> naga::ResourceBinding {
    naga::ResourceBinding { group, binding }
}

fn write_msl(
    def: &EffectDef,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String> {
    use naga::back::msl;

    let mut resources = msl::BindingMap::default();
    resources.insert(
        resource(slots::GROUP, slots::GLOBALS),
        msl::BindTarget {
            buffer: Some(slots::MSL_GLOBALS_BUFFER),
            ..Default::default()
        },
    );
    resources.insert(
        resource(slots::GROUP, slots::INSTANCES),
        msl::BindTarget {
            buffer: Some(slots::MSL_INSTANCES_BUFFER),
            ..Default::default()
        },
    );
    resources.insert(
        resource(slots::SOURCE_GROUP, slots::SOURCE_TEXTURE),
        msl::BindTarget {
            texture: Some(slots::MSL_SOURCE_TEXTURE),
            ..Default::default()
        },
    );
    // Written into the shader rather than bound, so the Metal renderer holds no
    // sampler state and the pipeline looks like every other one GPUI drives.
    resources.insert(
        resource(slots::SOURCE_GROUP, slots::SOURCE_SAMPLER),
        msl::BindTarget {
            sampler: Some(msl::BindSamplerTarget::Inline(0)),
            ..Default::default()
        },
    );

    let entry_point_resources = msl::EntryPointResources {
        resources,
        ..Default::default()
    };
    let mut per_entry_point_map = msl::EntryPointResourceMap::default();
    for entry in [VERTEX_ENTRY, FRAGMENT_ENTRY] {
        per_entry_point_map.insert(entry.to_string(), entry_point_resources.clone());
    }

    let options = msl::Options {
        // naga defaults to MSL 1.0, which has no `instance_id` attribute, so
        // the instanced quad every effect draws will not translate. 2.1 ships
        // with macOS 10.14 and is older than anything GPUI runs on.
        lang_version: (2, 1),
        per_entry_point_map,
        inline_samplers: vec![msl::sampler::InlineSampler {
            address: [msl::sampler::Address::ClampToBorder; 3],
            border_color: msl::sampler::BorderColor::TransparentBlack,
            mag_filter: msl::sampler::Filter::Linear,
            min_filter: msl::sampler::Filter::Linear,
            ..Default::default()
        }],
        ..Default::default()
    };
    let (source, _) = msl::write_string(module, info, &options, &Default::default())
        .with_context(|| format!("writing Metal shader for effect `{}`", def.name))?;
    Ok(source)
}

fn write_hlsl(
    def: &EffectDef,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String> {
    use naga::back::hlsl;

    let mut binding_map = hlsl::BindingMap::default();
    binding_map.insert(
        resource(slots::GROUP, slots::GLOBALS),
        hlsl::BindTarget {
            space: 0,
            register: slots::HLSL_GLOBALS_REGISTER,
            ..Default::default()
        },
    );
    binding_map.insert(
        resource(slots::GROUP, slots::INSTANCES),
        hlsl::BindTarget {
            space: 0,
            register: slots::HLSL_INSTANCES_REGISTER,
            ..Default::default()
        },
    );
    binding_map.insert(
        resource(slots::SOURCE_GROUP, slots::SOURCE_TEXTURE),
        hlsl::BindTarget {
            space: 0,
            register: slots::HLSL_SOURCE_REGISTER,
            ..Default::default()
        },
    );

    let options = hlsl::Options {
        // Direct3D 11 is the floor. Targeting anything higher would let an
        // effect compile here and fail on Windows.
        shader_model: hlsl::ShaderModel::V5_0,
        binding_map,
        ..Default::default()
    };

    let mut source = String::new();
    let mut writer = hlsl::Writer::new(&mut source, &options);
    writer
        .write(module, info, None)
        .with_context(|| format!("writing HLSL for effect `{}`", def.name))?;
    flatten_sampler_heap(&mut source, def.name)?;
    Ok(source)
}

/// Rewrite naga's Direct3D 12 sampler heap into a plain Shader Model 5.0 binding.
///
/// naga emits every sampler as an indirection through two 2048-entry heaps and a
/// per-group index buffer, addressed with register spaces. Spaces are Shader
/// Model 5.1 syntax, so fxc rejects the result even when asked for 5.0, and
/// naga offers no option to turn it off (gfx-rs/wgpu#8120).
///
/// The indirection exists so a bind group can hold an arbitrary number of
/// dynamically indexed samplers. The effect ABI holds exactly one, at a fixed
/// slot, never comparison and never indexed, so the whole apparatus collapses to
/// a single declaration. That is what makes this rewrite narrow enough to trust:
/// it is not a general translation, it is the one shape our own preamble emits.
///
/// An error here means naga's output stopped looking the way this expects, which
/// a test catches rather than a customer.
fn flatten_sampler_heap(source: &mut String, name: &str) -> Result<()> {
    if !source.contains(SAMPLER_HEAP) {
        // No sampler in this effect, so nothing to flatten.
        return Ok(());
    }

    let mut binding = None;
    let mut rewritten = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        // The heaps and the index buffer exist only to serve the indirection.
        if trimmed.starts_with("SamplerState nagaSamplerHeap")
            || trimmed.starts_with("SamplerComparisonState nagaComparisonSamplerHeap")
            || trimmed.contains("SamplerIndexArray :")
        {
            continue;
        }
        // `static const SamplerState x = nagaSamplerHeap[...];` becomes the
        // declaration of `x` itself.
        if let Some(rest) = trimmed.strip_prefix("static const SamplerState ")
            && let Some((sampler, _)) = rest.split_once(" = ")
            && rest.contains(SAMPLER_HEAP)
        {
            binding = Some(sampler.to_string());
            rewritten.push_str(&format!(
                "SamplerState {sampler} : register(s{});\n",
                slots::HLSL_SOURCE_SAMPLER_REGISTER
            ));
            continue;
        }
        rewritten.push_str(line);
        rewritten.push('\n');
    }

    let Some(binding) = binding else {
        bail!(
            "effect `{name}` uses a sampler, but naga's heap did not have the shape this \
             rewrites; see gfx-rs/wgpu#8120"
        );
    };
    if rewritten.contains(SAMPLER_HEAP) || rewritten.contains(", space") {
        bail!(
            "effect `{name}` still references the sampler heap after flattening `{binding}`; \
             naga's output has changed shape"
        );
    }

    *source = rewritten;
    Ok(())
}

/// The variable naga names its standard sampler heap.
const SAMPLER_HEAP: &str = "nagaSamplerHeap";

/// Compile one entry point the way the Direct3D renderer will.
///
/// Lives here rather than in the renderer because it is a statement about the
/// ABI, not about a device: it needs no adapter, no swap chain and no window,
/// which is exactly what lets a test call it.
#[cfg(windows)]
pub fn compile_hlsl(
    source: &str,
    name: &str,
    entry: &str,
    profile: &str,
) -> Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::core::PCSTR;

    let entry = std::ffi::CString::new(entry)?;
    let profile = std::ffi::CString::new(profile)?;
    let mut code = None;
    let mut errors = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr() as *const std::ffi::c_void,
            source.len(),
            None,
            None,
            None,
            PCSTR::from_raw(entry.as_ptr() as *const u8),
            PCSTR::from_raw(profile.as_ptr() as *const u8),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if result.is_err() {
        let detail = errors
            .map(|errors| unsafe {
                std::ffi::CStr::from_ptr(errors.GetBufferPointer() as *const i8)
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| format!("{result:?}"));
        bail!("effect `{name}` did not compile: {detail}");
    }
    code.context("D3DCompile reported success without producing bytecode")
}

/// Translate an effect and put it through fxc, as building a pipeline would.
///
/// [`translate`] proves naga emits HLSL. It cannot prove fxc accepts it, and
/// the two disagree often enough to matter — a register space, a resource
/// limit, an intrinsic that Shader Model 5.0 does not have. That disagreement
/// surfaces at pipeline creation, on a customer's machine, as a logged line and
/// a rectangle that never draws.
///
/// This is the same compiler call the renderer makes, so a pass here means the
/// renderer will get bytecode too.
#[cfg(windows)]
pub fn validate_direct3d(def: &EffectDef) -> Result<()> {
    let source = translate(def, ShaderTarget::Hlsl)?;
    for (entry, profile) in [(VERTEX_ENTRY, "vs_5_0"), (FRAGMENT_ENTRY, "ps_5_0")] {
        compile_hlsl(&source, def.name, entry, profile)
            .with_context(|| format!("compiling `{entry}` for effect `{}`", def.name))?;
    }
    Ok(())
}

/// Put every registered effect through fxc. See [`validate_direct3d`].
#[cfg(windows)]
pub fn validate_all_direct3d() -> Result<()> {
    let mut failures = Vec::new();
    for (_, def) in registered() {
        if let Err(error) = validate_direct3d(&def) {
            failures.push(format!("{error:#}"));
        }
    }
    if !failures.is_empty() {
        bail!(
            "{} effects were rejected by fxc:\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
    Ok(())
}

/// Translate every registered effect for every backend.
///
/// Call this from a test. A shader that fails to translate is a black rectangle
/// on a customer's machine and a stack trace nowhere, so the only good place to
/// find out is `cargo test`, on any platform, for all three backends at once.
pub fn validate_all() -> Result<()> {
    let mut failures = Vec::new();
    for (_, def) in registered() {
        for target in [ShaderTarget::Wgsl, ShaderTarget::Msl, ShaderTarget::Hlsl] {
            if let Err(error) = translate(&def, target) {
                failures.push(format!("{:?}: {error:#}", target));
            }
        }
    }
    if !failures.is_empty() {
        bail!("{} effect translations failed:\n{}", failures.len(), failures.join("\n\n"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest effect that touches every part of the ABI: a parameter of
    /// each kind, and a read of the source.
    const PROBE: EffectDef = EffectDef {
        name: "gpui-abi-probe",
        wgsl: "fn effect(input: EffectInput) -> vec4<f32> {
                   let sampled = source(input.uv);
                   let mixed = mix(sampled, tint(input), amount(input));
                   return vec4<f32>(to_encoded(to_linear(mixed.rgb)), mixed.a);
               }",
        parameters: &[
            Parameter {
                name: "amount",
                kind: ParameterKind::Scalar,
            },
            Parameter {
                name: "tint",
                kind: ParameterKind::Color,
            },
        ],
    };

    #[test]
    fn the_abi_translates_for_every_backend() {
        for target in [ShaderTarget::Wgsl, ShaderTarget::Msl, ShaderTarget::Hlsl] {
            translate(&PROBE, target).unwrap_or_else(|error| panic!("{target:?}: {error:#}"));
        }
    }

    #[test]
    fn metal_gets_its_sampler_written_into_the_shader() {
        // The Metal renderer binds no sampler state, exactly as GPUI's own
        // Metal pipelines bind none. If naga stops inlining it, sampling
        // silently returns nothing rather than failing to compile.
        let source = translate(&PROBE, ShaderTarget::Msl).unwrap();
        assert!(
            source.contains("constexpr metal::sampler") || source.contains("constexpr sampler"),
            "no inline sampler in the Metal output:\n{source}"
        );
        assert!(
            source.contains("filter::linear"),
            "the inline sampler is not filtered:\n{source}"
        );
        assert!(
            source.contains("clamp_to_border"),
            "the inline sampler does not stop at the capture's edge:\n{source}"
        );
        // MSL's default border is transparent black, so naga writes no
        // `border_color` at all for the one we want. Naming an opaque one is
        // the only way the border could be wrong.
        assert!(
            !source.contains("border_color::opaque"),
            "the inline sampler reads an opaque border outside the capture:\n{source}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn fxc_accepts_the_abi() {
        // The rest of the HLSL tests read naga's output as text. This one hands
        // it to the compiler that actually decides, which is the only thing
        // that can tell a black rectangle from a working effect.
        validate_direct3d(&PROBE).unwrap();
    }

    #[test]
    fn no_hlsl_reaches_fxc_with_a_register_space() {
        // Register spaces are Shader Model 5.1 syntax and Direct3D 11 compiles
        // 5.0, so one surviving `flatten_sampler_heap` is a shader that builds
        // in a test and fails on a customer's machine.
        let source = translate(&PROBE, ShaderTarget::Hlsl).unwrap();
        assert!(!source.contains(", space"), "a register space survived:\n{source}");
        assert!(!source.contains(SAMPLER_HEAP), "the sampler heap survived:\n{source}");
        assert!(
            source.contains(&format!(
                "register(s{})",
                slots::HLSL_SOURCE_SAMPLER_REGISTER
            )),
            "the flattened sampler is not at the register the renderer binds:\n{source}"
        );
    }

    #[test]
    fn the_shader_and_the_cpu_agree_on_the_instance_layout() {
        // WGSL and Rust apply different alignment rules, so the two structs can
        // drift apart with both sides still compiling. The effect then reads
        // its parameters out of padding, which looks like a broken shader.
        assert_eq!(
            shader_instance_size().unwrap() as usize,
            std::mem::size_of::<EffectInstance>()
        );
    }

    #[test]
    fn a_shader_that_calls_an_undeclared_parameter_fails() {
        let wrong = EffectDef {
            name: "gpui-abi-probe-undeclared",
            wgsl: "fn effect(input: EffectInput) -> vec4<f32> { return vec4<f32>(nope(input)); }",
            parameters: &[],
        };
        let error = translate(&wrong, ShaderTarget::Wgsl).unwrap_err();
        assert!(
            format!("{error:#}").contains("gpui-abi-probe-undeclared"),
            "the error does not name the effect: {error:#}"
        );
    }

    #[test]
    fn an_effect_that_wants_more_slots_than_exist_is_refused() {
        const TOO_MANY: &[Parameter] = &[Parameter {
            name: "colour",
            kind: ParameterKind::Color,
        }; PARAM_COUNT];
        let greedy = EffectDef {
            name: "gpui-abi-probe-greedy",
            wgsl: "fn effect(input: EffectInput) -> vec4<f32> { return colour0(input); }",
            parameters: TOO_MANY,
        };
        let error = translate(&greedy, ShaderTarget::Wgsl).unwrap_err();
        assert!(
            format!("{error:#}").contains("parameter slots"),
            "unexpected error: {error:#}"
        );
    }
}
