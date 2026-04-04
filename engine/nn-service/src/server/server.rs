use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tokio::sync::mpsc;
use crate::backends::Backend;
use alpha_cc_nn::proto::prediction_service_server::PredictionServiceServer;
use alpha_cc_nn::proto::management_service_server::ManagementServiceServer;
use crate::server::config::ServerConfig;
use crate::server::management::ManagementServiceImpl;
use crate::server::stages;
use crate::server::service::NNService;
use crate::server::types::{PendingPrediction, PipelineItem, StateBytes};


/// The main server. Construct with [`PredictServer::new`], then call
/// [`PredictServer::serve`] to start listening.
///
/// Generic over `B: Backend`, which defines the encode → inference → decode
/// → respond pipeline and its intermediate types.
pub struct PredictServer<B: Backend> {
    config: ServerConfig,
    backend: Arc<B>,
    static_mode: bool,
}

impl<B: Backend> PredictServer<B> {
    /// Create a new server with the given backend.
    pub fn new(config: ServerConfig, backend: B, static_mode: bool) -> Self {
        Self {
            config,
            backend: Arc::new(backend),
            static_mode,
        }
    }

    /// Clone of the backend Arc, for use by the reloader or other subsystems.
    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    /// Start serving on the given address (e.g. `"[::]:50051"`).
    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = addr.parse()?;

        // Build a routing table: model_id -> pipeline's submit_tx
        let mut route_map: HashMap<u32, mpsc::Sender<PendingPrediction>> = HashMap::new();

        for pipeline_cfg in &self.config.pipelines {
            let intake = pipeline_cfg.pipeline.intake_buffer;
            let outtake = pipeline_cfg.pipeline.outtake_buffer;

            let (submit_tx, submit_rx) = mpsc::channel(pipeline_cfg.batcher.channel_buffer);
            let (encoder_tx, encoder_rx) = mpsc::channel::<PipelineItem<Vec<StateBytes>>>(intake);
            let (inference_tx, inference_rx) = mpsc::channel::<PipelineItem<B::Encoded>>(intake);
            let (decoder_tx, decoder_rx) = mpsc::channel::<PipelineItem<B::Inferred>>(outtake);
            let (responder_tx, responder_rx) = mpsc::channel::<PipelineItem<Vec<(Vec<u8>, f32)>>>(outtake);

            let current_wait_us = Arc::new(AtomicU64::new(pipeline_cfg.batcher.max_wait.as_micros() as u64));
            tokio::spawn(stages::run_batcher(pipeline_cfg.batcher.clone(), submit_rx, encoder_tx, current_wait_us.clone()));
            tokio::spawn(stages::run_encoder(self.backend.clone(), encoder_rx, inference_tx));
            tokio::spawn(stages::run_inference(self.backend.clone(), inference_rx, decoder_tx));
            tokio::spawn(stages::run_decoder(self.backend.clone(), decoder_rx, responder_tx));
            tokio::spawn(stages::run_responder(self.backend.clone(), responder_rx, pipeline_cfg.model_ids.clone(), current_wait_us));

            for &model_id in &pipeline_cfg.model_ids {
                route_map.insert(model_id, submit_tx.clone());
            }
            let ids: Vec<_> = pipeline_cfg.model_ids.iter().map(|id| id.to_string()).collect();
            log::info!(
                "Pipeline spawned for model_ids=[{}] (batch_size={}, wait={:?}-{:?})",
                ids.join(", "),
                pipeline_cfg.batcher.max_batch_size,
                pipeline_cfg.batcher.min_wait,
                pipeline_cfg.batcher.max_wait,
            );
        }

        let backend_for_svc = self.backend.clone();
        let svc = NNService {
            route_map: Arc::new(route_map),
            model_id_ready: move |model_id: u32| {
                backend_for_svc.model_store().load(model_id as usize).is_some()
            },
        };

        let mgmt_svc = ManagementServiceImpl::new(
            self.backend.clone(),
            self.config.clone(),
            self.static_mode,
        );

        log::info!("PredictServer listening on {addr}");

        tonic::transport::Server::builder()
            .add_service(PredictionServiceServer::new(svc))
            .add_service(ManagementServiceServer::new(mgmt_svc).max_decoding_message_size(256 * 1024 * 1024))
            .serve(addr)
            .await?;

        Ok(())
    }
}
