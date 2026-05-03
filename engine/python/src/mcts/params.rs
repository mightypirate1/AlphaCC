use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "ImprovedHalvingParams", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyImprovedHalvingParams {
    #[pyo3(get, set)] pub c_visit: f32,
    #[pyo3(get, set)] pub c_scale: f32,
    #[pyo3(get, set)] pub all_at_least_once: bool,
    #[pyo3(get, set)] pub base_count: usize,
    #[pyo3(get, set)] pub floor_count: usize,
    #[pyo3(get, set)] pub keep_frac: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImprovedHalvingParams {
    #[new]
    #[pyo3(signature = (c_visit=50.0, c_scale=1.0, all_at_least_once=false, base_count=16, floor_count=5, keep_frac=0.5))]
    fn new(c_visit: f32, c_scale: f32, all_at_least_once: bool, base_count: usize, floor_count: usize, keep_frac: f32) -> Self {
        Self { c_visit, c_scale, all_at_least_once, base_count, floor_count, keep_frac }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "PuctFreeParams", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyPuctFreeParams {
    #[pyo3(get, set)] pub c_puct_init: f32,
    #[pyo3(get, set)] pub c_puct_base: f32,
    #[pyo3(get, set)] pub dirichlet_weight: f32,
    #[pyo3(get, set)] pub dirichlet_alpha: f32,
    #[pyo3(get, set)] pub temperature: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPuctFreeParams {
    #[new]
    #[pyo3(signature = (c_puct_init=2.0, c_puct_base=10000.0, dirichlet_weight=0.15, dirichlet_alpha=0.15, temperature=1.0))]
    fn new(c_puct_init: f32, c_puct_base: f32, dirichlet_weight: f32, dirichlet_alpha: f32, temperature: f32) -> Self {
        Self { c_puct_init, c_puct_base, dirichlet_weight, dirichlet_alpha, temperature }
    }
}
