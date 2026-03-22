use std::sync::Arc;
use tokio::sync::mpsc;
use crate::nn::backends::Backend;
use crate::nn::proto::prediction_service_server::PredictionServiceServer;
use crate::nn::server::config::ServerConfig;
use crate::nn::server::stages;
use crate::nn::server::service::NNService;
use crate::nn::server::types::{PipelineItem, StateBytes};


/// The main server. Construct with [`PredictServer::new`], then call
/// [`PredictServer::serve`] to start listening.
///
/// Generic over `B: Backend`, which defines the encode → inference → decode
/// → respond pipeline and its intermediate types.
pub struct PredictServer<B: Backend> {
    config: ServerConfig,
    backend: Arc<B>,
}

impl<B: Backend> PredictServer<B> {
    /// Create a new server with the given backend.
    pub fn new(config: ServerConfig, backend: B) -> Self {
        Self {
            config,
            backend: Arc::new(backend),
        }
    }

    /// Clone of the backend Arc, for use by the reloader or other subsystems.
    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    /// Start serving on the given address (e.g. `"[::]:50051"`).
    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = addr.parse()?;
        let capacity = self.config.pipeline.buffer_size;

        // grpc -> batcher -> encoder -> inference -> decoder -> responder
        let (submit_tx, submit_rx) = mpsc::channel(self.config.batcher.channel_buffer);
        let (encoder_tx, encoder_rx) = mpsc::channel::<PipelineItem<Vec<StateBytes>>>(capacity);
        let (inference_tx, inference_rx) = mpsc::channel::<PipelineItem<B::Encoded>>(capacity);
        let (decoder_tx, decoder_rx) = mpsc::channel::<PipelineItem<B::Inferred>>(capacity);
        let (responder_tx, responder_rx) = mpsc::channel::<PipelineItem<Vec<(Vec<u8>, f32)>>>(capacity);

        tokio::spawn(stages::run_batcher(self.config.batcher, submit_rx, encoder_tx));
        tokio::spawn(stages::run_encoder(self.backend.clone(), encoder_rx, inference_tx));
        tokio::spawn(stages::run_inference(self.backend.clone(), inference_rx, decoder_tx));
        tokio::spawn(stages::run_decoder(self.backend.clone(), decoder_rx, responder_tx));
        tokio::spawn(stages::run_responder(self.backend.clone(), responder_rx));

        let backend_for_svc = self.backend.clone();
        let svc = NNService {
            submit_tx,
            model_id_ready: move |model_id: u32| {
                backend_for_svc.model_store().load(model_id as usize).is_some()
            },
        };

        println!("PredictServer listening on {addr}");

        tonic::transport::Server::builder()
            .add_service(PredictionServiceServer::new(svc))
            .serve(addr)
            .await?;

        Ok(())
    }
}
