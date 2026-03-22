use crate::cc::game::hexcoord::HexCoord;

#[cfg(feature = "extension-module")]
extern crate pyo3;


#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", from_py_object, get_all))]
#[derive(Clone)]
pub struct Move {
    pub from_coord: HexCoord,
    pub to_coord: HexCoord,
    pub path: Vec<HexCoord>,
}

#[cfg(feature = "extension-module")]
#[pyo3::prelude::pymethods]
impl Move {
    pub fn __repr__(&self) -> String {
        let fc = self.from_coord;
        let tc = self.to_coord;
        format!("Move[{} -> {}]", fc.repr(), tc.repr())
    }
}