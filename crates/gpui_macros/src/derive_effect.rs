use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input};

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

    // Every field is described and written through `Parameter`, so this macro
    // never has to recognise a type. It cannot: a derive sees the spelling of a
    // field's type and nothing else, and matching on the spelling accepts an
    // alias or a same-named type from elsewhere and then reads it as something
    // it is not. Deferring to the trait turns that into a bound error on the
    // field, and lets an application add a type without touching GPUI.
    let mut declarations = Vec::new();
    let mut writes = Vec::new();
    let mut widths = Vec::new();
    let mut agreements = Vec::new();
    for field in &fields.named {
        let Some(identifier) = &field.ident else {
            continue;
        };
        let literal = LitStr::new(&identifier.to_string(), identifier.span());
        let ty = &field.ty;

        declarations.push(quote! {
            gpui::effect::ParameterDef {
                name: #literal,
                kind: <#ty as gpui::effect::Parameter>::KIND,
            }
        });
        writes.push(quote! { out.put(&self.#identifier); });
        widths.push(quote! {
            <#ty as gpui::effect::Parameter>::KIND.slots()
        });
        // A parameter that claims one width and occupies another shifts every
        // field after it, which reads as an effect ignoring its own settings.
        // Checked per field with a concrete type, so the message names it.
        agreements.push(quote! {
            const _: () = assert!(
                <#ty as gpui::effect::Parameter>::KIND.slots()
                    == <<#ty as gpui::effect::Parameter>::Slots as gpui::effect::Slots>::WIDTH,
                concat!(
                    "parameter `",
                    stringify!(#identifier),
                    "` occupies a different number of slots than its kind claims",
                ),
            );
        });
    }

    quote! {
        #(#agreements)*

        // The budget is fixed and every width is a constant, so overrunning it
        // is something the compiler can know. Left to `translate`, the same
        // mistake is an effect that registers, fails at startup and draws
        // nothing.
        const _: () = {
            let mut total = 0usize;
            #( total += #widths; )*
            assert!(
                total <= gpui::effect::PARAM_COUNT,
                concat!(
                    "effect `",
                    stringify!(#type_name),
                    "` needs more parameter slots than an effect has",
                ),
            );
        };

        impl gpui::effect::Effect for #type_name {
            const NAME: &'static str = #name;
            const SOURCE: &'static str = include_str!(#source);
            const PARAMETERS: &'static [gpui::effect::ParameterDef] = &[#(#declarations),*];

            fn write(&self, out: &mut gpui::effect::Params) {
                #(#writes)*
            }
        }
    }
    .into()
}

fn error(span: proc_macro2::Span, message: &str) -> TokenStream {
    syn::Error::new(span, message).into_compile_error().into()
}
