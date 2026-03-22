pub mod backends;
pub mod client;
pub mod io;
pub mod reloads;
pub mod server;

/// Generated protobuf types and gRPC service definitions.
pub mod proto {
    tonic::include_proto!("predict");
}
