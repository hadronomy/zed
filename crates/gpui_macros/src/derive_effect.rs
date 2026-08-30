use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, Type, parse_macro_input, spanned::Spanned};

pub fn derive_effect(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let type_name = &ast.ident;

    let mut name = None;
    let mut source = None;
    for attribute in ast.attrs.iter().filter(|a| a.path().is_ident("effect")) {
        if let Err(error) = attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                name = Some(meta.value()?.parse::<LitStr>()?);
                Ok(())
            } else if meta.path.is_ident("source") {
                source = Some(meta.value()?.parse::<LitStr>()?);
                Ok(())
            } else {
                Err(meta.error("expected `name` or `source`"))
            }
        }) {
            return error.into_compile_error().into();
        }
    }

    let (Some(name), Some(source)) = (name, source) else {
        return error(
            ast.ident.span(),
            "expected #[effect(name = \"...\", source = \"...\")]",
        );
    };
    let Data::Struct(data) = &ast.data else {
        return error(ast.ident.span(), "an effect must be a struct");
    };
    let Fields::Named(fields) = &data.fields else {
        return error(ast.ident.span(), "an effect must have named fields");
    };

    let mut parameters = Vec::new();
    let mut writes = Vec::new();
    let mut slot = 0usize;
    for field in &fields.named {
        let Some(identifier) = &field.ident else {
            continue;
        };
        let literal = LitStr::new(&identifier.to_string(), identifier.span());

        match type_name_of(&field.ty).as_deref() {
            Some("f32") => {
                parameters.push(quote! {
                    gpui::effect::Parameter {
                        name: #literal,
                        kind: gpui::effect::ParameterKind::Scalar,
                    }
                });
                writes.push(quote! { params[#slot] = self.#identifier; });
                slot += 1;
            }
            Some("Hsla") => {
                let (hue, saturation, lightness, alpha) =
                    (slot, slot + 1, slot + 2, slot + 3);
                parameters.push(quote! {
                    gpui::effect::Parameter {
                        name: #literal,
                        kind: gpui::effect::ParameterKind::Color,
                    }
                });
                writes.push(quote! {
                    params[#hue] = self.#identifier.h;
                    params[#saturation] = self.#identifier.s;
                    params[#lightness] = self.#identifier.l;
                    params[#alpha] = self.#identifier.a;
                });
                slot += 4;
            }
            _ => {
                return error(
                    field.ty.span(),
                    "an effect parameter must be `f32` or `Hsla`",
                );
            }
        }
    }

    quote! {
        impl gpui::effect::Effect for #type_name {
            const NAME: &'static str = #name;
            const SOURCE: &'static str = include_str!(#source);
            const PARAMETERS: &'static [gpui::effect::Parameter] = &[#(#parameters),*];

            fn params(&self) -> [f32; gpui::effect::PARAM_COUNT] {
                let mut params = [0.0; gpui::effect::PARAM_COUNT];
                #(#writes)*
                params
            }
        }
    }
    .into()
}

fn type_name_of(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) => Some(path.path.segments.last()?.ident.to_string()),
        _ => None,
    }
}

fn error(span: proc_macro2::Span, message: &str) -> TokenStream {
    syn::Error::new(span, message).into_compile_error().into()
}
