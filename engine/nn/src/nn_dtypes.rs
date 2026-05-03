/// Quantized value in `[-1,1]` stored as i16 (Q1.15)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bitcode::Encode, bitcode::Decode)]
pub struct NNQuantizedValue(i16);


impl NNQuantizedValue {
    pub const MIN: f32 = -1.0;
    pub const MAX: f32 = 1.0;
    pub const SCALE: f32 = i16::MAX as f32;
    pub const INV_SCALE: f32 = 1.0 / (i16::MAX as f32);

    #[inline]
    pub fn quantize(v: f32) -> Self {
        let c = v.clamp(-1.0, 1.0);
        // round() gives symmetric rounding for negatives.
        NNQuantizedValue((c * Self::SCALE).round() as i16)
    }

    #[inline]
    pub fn dequantize(self) -> f32 {
        (self.0 as f32) * Self::INV_SCALE
    }

    #[inline]
    pub fn raw(self) -> i16 { self.0 }

    #[inline]
    pub fn from_raw(raw: i16) -> Self { NNQuantizedValue(raw) }

    #[inline]
    pub fn max_abs_error() -> f32 { 0.5 / Self::SCALE }
}


impl From<f32> for NNQuantizedValue {
    #[inline]
    fn from(v: f32) -> Self { NNQuantizedValue::quantize(v) }
}

impl From<NNQuantizedValue> for f32 {
    #[inline]
    fn from(q: NNQuantizedValue) -> f32 { q.dequantize() }
}

impl From<i16> for NNQuantizedValue {
    #[inline]
    fn from(raw: i16) -> Self { NNQuantizedValue::from_raw(raw) }
}

impl From<NNQuantizedValue> for i16 {
    #[inline]
    fn from(q: NNQuantizedValue) -> i16 { q.raw() }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng, RngExt};

    #[test]
    fn value_roundtrip_fixed_points() {
        for &x in &[-1.0, -0.777, -0.001, 0.0, 0.42, 0.9999, 1.0] {
            let q = NNQuantizedValue::from(x);
            let r = q.dequantize();
            assert!((x - r).abs() <= NNQuantizedValue::max_abs_error() + 1e-6, "x={x} r={r}");
        }
    }

    #[test]
    fn value_clamping() {
        assert_eq!(NNQuantizedValue::quantize(-2.0).dequantize(), -1.0);
        assert_eq!(NNQuantizedValue::quantize(2.0).dequantize(), 1.0);
    }

    #[test]
    fn random_value_error_bound() {
        let mut rng = StdRng::seed_from_u64(456);
        for _ in 0..10_000 {
            let x = rng.random_range(-1.0..=1.0);
            let r = NNQuantizedValue::from(x).dequantize();
            assert!((x - r).abs() <= NNQuantizedValue::max_abs_error() + 1e-6);
        }
    }
}
