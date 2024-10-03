#[derive(Clone)]
pub struct MCTSParams {
    pub gamma: f32,
    pub dirichlet_weight: f32,
    pub dirichlet_alpha: f32,
    pub c_puct_init: f32,
    pub c_puct_base: f32,
}
