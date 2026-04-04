pub mod nn_dtypes;
pub mod nn_pred;
pub mod inference_utils;
pub mod io;
pub mod client;
pub mod nn_remote;
pub mod prediction_client;

pub mod proto {
    tonic::include_proto!("predict");
}

pub use nn_dtypes::{NNQuantizedPi, NNQuantizedValue};
pub use nn_pred::NNPred;
pub use nn_remote::NNRemote;
pub use prediction_client::{PredictionClient, FetchStats};

#[cfg(feature = "extension-module")]
pub use inference_utils::{preds_from_logits, build_inference_request};
