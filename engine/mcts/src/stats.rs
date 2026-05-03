#[derive(Clone, Default)]
pub struct SearchStats {
    pub prior_entropy: f32,
    pub target_entropy: f32,
    pub logit_std: f32,
    pub sigma_q_std: f32,
    pub kl_prior_posterior: f32,
    pub kl_posterior_prior: f32,
}

impl SearchStats {
    pub fn compute(
        prior_pi: &[f32],
        posterior_pi: &[f32],
        pi_logits: &[f32],
        sigma_qs: &[f32],
    ) -> Self {
        Self {
            prior_entropy: entropy(prior_pi),
            target_entropy: entropy(posterior_pi),
            logit_std: std_dev(pi_logits),
            sigma_q_std: std_dev(sigma_qs),
            kl_prior_posterior: kl_divergence(prior_pi, posterior_pi),
            kl_posterior_prior: kl_divergence(posterior_pi, prior_pi),
        }
    }
}

fn entropy(probs: &[f32]) -> f32 {
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    variance.sqrt()
}

/// KL(p || q) = Σ p_i * log(p_i / q_i). Terms with p_i = 0 contribute 0.
/// q_i is clamped to a small epsilon to avoid div-by-zero / log(0) under softmax underflow.
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-12;
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi.max(EPS)).ln())
        .sum()
}
