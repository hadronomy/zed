use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input};

pub fn derive_shader_enum(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let type_name = &ast.ident;

    let Data::Enum(data) = &ast.data else {
        return error(ast.ident.span(), "a shader enum must be an enum");
    };

    let mut variants = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return error(
                variant.ident.span(),
                "a shader enum's variants carry no data: one slot holds a number, and there is \
                 nowhere to put anything else",
            );
        }
        let ident = &variant.ident;
        let literal = LitStr::new(&ident.to_string(), ident.span());
        // `as u32` reads whatever the enum declares, so an explicit
        // discriminant is honoured and the shader's constant follows it.
        variants.push(quote! {
            gpui::effect::Variant {
                name: #literal,
                value: #type_name::#ident as u32,
            }
        });
    }

    quote! {
        impl gpui::effect::ShaderEnum for #type_name {
            const VARIANTS: &'static [gpui::effect::Variant] = &[#(#variants),*];

            fn discriminant(self) -> u32 {
                self as u32
            }
        }

        // Written out rather than blanket-implemented over `ShaderEnum`: a
        // blanket impl overlaps `impl Parameter for f32`, because nothing can
        // prove `f32` is not a shader enum. serde has the same constraint and
        // answers it the same way.
        impl gpui::effect::Parameter for #type_name {
            const KIND: gpui::effect::ParameterKind = gpui::effect::ParameterKind::Enum(
                <#type_name as gpui::effect::ShaderEnum>::VARIANTS,
            );
            type Slots = [f32; 1];

            fn slots(&self) -> Self::Slots {
                [gpui::effect::ShaderEnum::discriminant(*self) as f32]
            }
        }
    }
    .into()
}

fn error(span: proc_macro2::Span, message: &str) -> TokenStream {
    syn::Error::new(span, message).into_compile_error().into()
}
