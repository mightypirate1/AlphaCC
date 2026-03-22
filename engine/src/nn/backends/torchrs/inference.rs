use tch::{CModule, IValue, Tensor};

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
