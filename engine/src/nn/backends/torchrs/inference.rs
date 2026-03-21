use std::time::Duration;
use tch::{CModule, Device, IValue, Kind, Tensor};

pub fn nn_inference(nn: &CModule, tensor: Tensor) -> (Tensor, Tensor) {
    let _guard = tch::no_grad_guard();
    let output = nn.forward_is(&[IValue::Tensor(tensor)]).unwrap();
    match output {
        IValue::Tuple(vals) => {
            let pi = match &vals[0] { IValue::Tensor(t) => t.shallow_clone(), _ => panic!("expected tensor") };
            let v = match &vals[1] { IValue::Tensor(t) => t.shallow_clone(), _ => panic!("expected tensor") };
            (pi, v)
        }
        _ => panic!("expected tuple output from model"),
    }
}

pub fn fake_inference(tensor: Tensor, game_size: i64, device: Device) -> (Tensor, Tensor) {
    let batch_size = tensor.size()[0];
    let jitter = rand::RngExt::random_range(&mut rand::rng(), 3350..=3450);
    std::thread::sleep(Duration::from_micros(jitter));
    let s = game_size;
    let opts = (Kind::Float, device);
    let pi = Tensor::zeros([batch_size, s * s * s * s], opts);
    let v = Tensor::zeros([batch_size], opts);
    (pi, v)
}
