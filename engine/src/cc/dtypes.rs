// board
pub type BoardSize = u8;
pub type BoardContent = i8;
pub type HomeCapacity = u8;
pub type GameDuration = u16;
pub type EncBoard = Vec<u8>;
pub type BoardHash = u64;

// rollouts and nn
/// Quantized probability in `[0,1]` stored as u16 (Q0.16)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bincode::Encode, bincode::BorrowDecode)]
pub struct NNQuantizedPi(u16);

impl NNQuantizedPi {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 1.0;
    pub const SCALE: f32 = u16::MAX as f32;          // 65535.0
    pub const INV_SCALE: f32 = 1.0 / (u16::MAX as f32);

    #[inline]
    pub fn quantize(p: f32) -> Self {
        let c = p.clamp(0.0, 1.0);
        // Round to nearest (ties to +inf); acceptable for probabilities.
        NNQuantizedPi((c * Self::SCALE + 0.5) as u16)
    }

    #[inline]
    pub fn quantize_vec(src: &[f32]) -> Vec<NNQuantizedPi> {
        src.iter().copied().map(NNQuantizedPi::quantize).collect()
    }

    #[inline]
    pub fn dequantize(self) -> f32 {
        (self.0 as f32) * Self::INV_SCALE
    }

    #[inline]
    pub fn quantize_into(dst: &mut [NNQuantizedPi], src: &[f32]) {
        assert_eq!(dst.len(), src.len());
        for (d, &p) in dst.iter_mut().zip(src) {
            *d = NNQuantizedPi::quantize(p);
        }
    }

    #[inline]
    pub fn dequantize_into(dst: &mut [f32], src: &[NNQuantizedPi]) {
        assert_eq!(dst.len(), src.len());
        for (d, &q) in dst.iter_mut().zip(src) {
            *d = q.dequantize();
        }
    }

    #[inline]
    pub fn raw(self) -> u16 { self.0 }

    #[inline]
    pub fn from_raw(raw: u16) -> Self { NNQuantizedPi(raw) }

    #[inline]
    pub fn max_abs_error() -> f32 { 0.5 / Self::SCALE }
}


impl From<f32> for NNQuantizedPi {
    #[inline]
    fn from(v: f32) -> Self { NNQuantizedPi::quantize(v) }
}

impl From<NNQuantizedPi> for f32 {
    #[inline]
    fn from(q: NNQuantizedPi) -> f32 { q.dequantize() }
}

impl From<u16> for NNQuantizedPi {
    #[inline]
    fn from(raw: u16) -> Self { NNQuantizedPi::from_raw(raw) }
}

impl From<NNQuantizedPi> for u16 {
    #[inline]
    fn from(q: NNQuantizedPi) -> u16 { q.raw() }
}


/// Quantized value in `[-1,1]` stored as i16 (Q1.15)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bincode::Encode, bincode::BorrowDecode)]
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
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn pi_roundtrip_fixed_points() {
        for &x in &[0.0, 1e-6, 0.123456, 0.5, 0.99999, 1.0] {
            let q = NNQuantizedPi::from(x);
            let r = q.dequantize();
            assert!((x - r).abs() <= NNQuantizedPi::max_abs_error() + 1e-6, "x={x} r={r}");
        }
    }

    #[test]
    fn value_roundtrip_fixed_points() {
        for &x in &[-1.0, -0.777, -0.001, 0.0, 0.42, 0.9999, 1.0] {
            let q = NNQuantizedValue::from(x);
            let r = q.dequantize();
            assert!((x - r).abs() <= NNQuantizedValue::max_abs_error() + 1e-6, "x={x} r={r}");
        }
    }

    #[test]
    fn pi_clamping() {
        assert_eq!(NNQuantizedPi::quantize(-0.5).dequantize(), 0.0);
        assert_eq!(NNQuantizedPi::quantize(1.5).dequantize(), 1.0);
    }

    #[test]
    fn value_clamping() {
        assert_eq!(NNQuantizedValue::quantize(-2.0).dequantize(), -1.0);
        assert_eq!(NNQuantizedValue::quantize(2.0).dequantize(), 1.0);
    }

    #[test]
    fn random_pi_error_bound() {
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..10_000 {
            let x = rng.gen::<f32>();
            let r = NNQuantizedPi::from(x).dequantize();
            assert!((x - r).abs() <= NNQuantizedPi::max_abs_error() + 1e-6);
        }
    }

    #[test]
    fn random_value_error_bound() {
        let mut rng = StdRng::seed_from_u64(456);
        for _ in 0..10_000 {
            let x = rng.gen_range(-1.0..=1.0);
            let r = NNQuantizedValue::from(x).dequantize();
            assert!((x - r).abs() <= NNQuantizedValue::max_abs_error() + 1e-6);
        }
    }

    #[test]
    fn serialization_roundtrip() {
        use bincode::config::standard;
        let pis: Vec<f32> = (0..100).map(|i| (i as f32) / 99.0).collect();
        let quant = NNQuantizedPi::quantize_vec(&pis);
        let encoded = bincode::encode_to_vec(&quant, standard()).unwrap();
        let (decoded, _len): (Vec<NNQuantizedPi>, usize) = bincode::borrow_decode_from_slice(&encoded, standard()).unwrap();
        for (a,b) in quant.iter().zip(decoded.iter()) {
            assert_eq!(a.raw(), b.raw());
        }
    }
}
