pub mod backends;
pub mod client;
pub mod io;
pub mod nn_remote;
pub mod prediction_client;
pub mod server;
pub mod reloads;
pub mod db;

pub mod proto {
    tonic::include_proto!("predict");
}

pub use nn_remote::NNRemote;
pub use prediction_client::{PredictionClient, FetchStats};
