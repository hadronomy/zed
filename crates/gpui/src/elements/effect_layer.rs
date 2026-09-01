use crate::{
    AnyElement, App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, PaintEffect, Pixels, Window,
    effect::{self, EffectId},
};

/// Wrap `child` so an effect is applied to what it draws.
///
/// The child is painted into a texture of its own and composited through the
/// effect, which is what lets the effect blur, displace or tint the result
/// rather than paint pixels of its own. See [`Window::paint_effect_over`] for
/// what that costs and what it changes about the child.
pub fn effect_layer<E: effect::Effect>(effect: &E, child: impl IntoElement) -> EffectLayer {
    EffectLayer {
        effect: effect::register(E::definition()),
        params: effect::Params::of(effect),
        outset: Pixels::ZERO,
        corner_radii: Corners::default(),
        child: child.into_any_element(),
    }
}

/// An element that applies an effect to its child. See [`effect_layer`].
pub struct EffectLayer {
    effect: EffectId,
    params: effect::Params,
    outset: Pixels,
    corner_radii: Corners<Pixels>,
    child: AnyElement,
}

impl EffectLayer {
    /// Give the effect room to spread past the child, without moving the
    /// child's neighbours. See [`PaintEffect::outset`].
    pub fn outset(mut self, outset: Pixels) -> Self {
        self.outset = outset;
        self
    }

    /// Round the composited result, the way a quad's corners round.
    pub fn corner_radii(mut self, corner_radii: impl Into<Corners<Pixels>>) -> Self {
        self.corner_radii = corner_radii.into();
        self
    }
}

impl IntoElement for EffectLayer {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for EffectLayer {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (self.child.request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let effect = PaintEffect {
            bounds,
            outset: self.outset,
            corner_radii: self.corner_radii,
            effect: self.effect,
            params: self.params,
        };
        window.paint_effect_over(effect, |window| self.child.paint(window, cx));
    }
}
