use pyo3::prelude::*;
use pyo3_stub_gen_derive::gen_stub_pyclass;

#[gen_stub_pyclass]
#[pyclass(name = "FetchStats", module = "alpha_cc_engine", get_all)]
pub struct PyFetchStats {
    pub total_fetch_time_us: u64,
    pub total_fetches: u32,
}

impl From<alpha_cc_nn::FetchStats> for PyFetchStats {
    fn from(fs: alpha_cc_nn::FetchStats) -> Self {
        PyFetchStats {
            total_fetch_time_us: fs.total_fetch_time_us,
            total_fetches: fs.total_fetches,
        }
    }
}
