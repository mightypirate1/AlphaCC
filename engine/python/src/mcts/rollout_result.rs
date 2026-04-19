use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "RolloutResult", module = "alpha_cc_engine")]
pub struct PyRolloutResult(pub alpha_cc_mcts::RolloutResult);

#[gen_stub_pymethods]
#[pymethods]
impl PyRolloutResult {
    #[new]
    #[pyo3(signature = (pi, value, mcts_wdl, greedy_backup_wdl=None))]
    fn new(pi: Vec<f32>, value: f32, mcts_wdl: [f32; 3], greedy_backup_wdl: Option<[f32; 3]>) -> Self {
        let greedy_backup_wdl = greedy_backup_wdl.unwrap_or(mcts_wdl);
        Self(alpha_cc_mcts::RolloutResult {
            pi, value, mcts_wdl, greedy_backup_wdl,
            search_stats: alpha_cc_mcts::SearchStats::default(),
        })
    }

    #[getter]
    fn pi<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(self.0.pi.clone()), py,
        )
    }

    #[getter]
    fn value(&self) -> f32 {
        self.0.value
    }

    #[getter]
    fn mcts_wdl(&self) -> [f32; 3] {
        self.0.mcts_wdl
    }

    #[getter]
    fn greedy_backup_wdl(&self) -> [f32; 3] {
        self.0.greedy_backup_wdl
    }

    #[getter]
    fn prior_entropy(&self) -> f32 {
        self.0.search_stats.prior_entropy
    }

    #[getter]
    fn target_entropy(&self) -> f32 {
        self.0.search_stats.target_entropy
    }

    #[getter]
    fn logit_std(&self) -> f32 {
        self.0.search_stats.logit_std
    }

    #[getter]
    fn sigma_q_std(&self) -> f32 {
        self.0.search_stats.sigma_q_std
    }

    #[getter]
    fn kl_prior_posterior(&self) -> f32 {
        self.0.search_stats.kl_prior_posterior
    }

    #[getter]
    fn kl_posterior_prior(&self) -> f32 {
        self.0.search_stats.kl_posterior_prior
    }
}
