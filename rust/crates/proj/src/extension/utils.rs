macro_rules! readonly_struct {
    ($name:ident, $($struct_doc:expr)+, $({$field:ident: $type:ty $(, $field_doc:expr)?}),*) => {
        $(#[doc=$struct_doc])+
        #[cfg_attr(feature = "serde", derive(serde::Serialize))]
        #[derive(Debug)]
        pub struct $name {
            $( $field: $type ),*
        }

        impl $name {
            // Constructor function to initialize the struct
            #[allow(dead_code)]
            pub fn new($($field: $type),*) -> Self {
                $name {
                    $( $field ),*
                }
            }

            // Getter methods for each field
            $(
                $(#[doc=$field_doc])?
                pub fn $field(&self) -> &$type {
                    &self.$field
                }
            )*
        }
    }
}

pub(crate) use readonly_struct;
impl crate::Proj {
    /// Panic if a `Proj` object is not CRS.
    pub fn assert_crs(&self) -> miette::Result<&Self> {
        if !self.is_crs() {
            miette::bail!("Proj object is not CRS.");
        }
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_assert_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        //is crs
        {
            let pj = ctx.clone().create("EPSG:4326")?;
            assert!(pj.assert_crs().is_ok());
        }
        //not crs
        {
            let pj = ctx.create("+proj=utm +zone=32 +datum=WGS84")?;
            assert!(pj.assert_crs().is_err());
        }
        Ok(())
    }
}
