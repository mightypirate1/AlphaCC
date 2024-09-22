use crate::cc::hexcoord::HexCoord;
extern crate pyo3;
use pyo3::prelude::*;


#[pyclass(module="alpha_cc_engine")]
#[derive(Clone)]
pub struct Move {
    #[pyo3(get)]
    pub from_coord: HexCoord, 
    #[pyo3(get)]
    pub to_coord: HexCoord,
    #[pyo3(get)]
    pub path: Vec<HexCoord>,
}

#[pymethods]
impl Move {
    pub fn __repr__(&self) -> String {
        let fc = self.from_coord;
        let tc = self.to_coord;
        format!("Move[{} -> {}]", fc.__repr__(), tc.__repr__())
    }
}