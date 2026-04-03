use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::nn::backends::{Backend, VersionedModel};
use crate::nn::proto::management_service_server::ManagementService;
use crate::nn::proto::{
    ChannelInfo, LoadModelRequest, LoadModelResponse, ServerInfoRequest, ServerInfoResponse,
};
use crate::nn::server::config::ServerConfig;

pub struct ManagementServiceImpl<B: Backend> {
    backend: Arc<B>,
    config: ServerConfig,
    static_mode: bool,
}

impl<B: Backend> ManagementServiceImpl<B> {
    pub fn new(backend: Arc<B>, config: ServerConfig, static_mode: bool) -> Self {
        Self { backend, config, static_mode }
    }
}

#[tonic::async_trait]
impl<B: Backend> ManagementService for ManagementServiceImpl<B> {
    async fn get_server_info(
        &self,
        _request: Request<ServerInfoRequest>,
    ) -> Result<Response<ServerInfoResponse>, Status> {
        let store = self.backend.model_store();
        let current = store.current_models();

        let mut channels = Vec::new();
        for id in 0..store.len() {
            channels.push(ChannelInfo {
                channel_id: id as u32,
                model_loaded: current.contains_key(&id),
                model_version: current.get(&id).copied().unwrap_or(0) as u32,
            });
        }

        let batch_sizes: Vec<u32> = self
            .config
            .pipelines
            .iter()
            .map(|p| p.batcher.max_batch_size as u32)
            .collect();

        Ok(Response::new(ServerInfoResponse {
            game_size: self.config.game_size as u32,
            channels,
            batch_sizes,
            static_mode: self.static_mode,
        }))
    }

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        if !self.static_mode {
            return Ok(Response::new(LoadModelResponse {
                success: false,
                error: "LoadModel is only available in static mode (server is in redis-polling mode)".into(),
            }));
        }

        let req = request.into_inner();
        let channel_id = req.channel_id as usize;

        if channel_id >= self.backend.model_store().len() {
            return Ok(Response::new(LoadModelResponse {
                success: false,
                error: format!(
                    "channel_id {} out of range (max {})",
                    channel_id,
                    self.backend.model_store().len() - 1
                ),
            }));
        }

        let model = match self.backend.model_from_bytes(&req.onnx_bytes) {
            Ok(m) => m,
            Err(e) => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: format!("failed to load model from bytes: {e}"),
                }));
            }
        };

        let model = match self.backend.compile_model(model) {
            Ok(m) => m,
            Err(e) => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: format!("failed to compile model: {e}"),
                }));
            }
        };

        self.backend.model_store().set(
            channel_id,
            VersionedModel { model, version: req.version as usize },
        );

        log::info!("Loaded model into channel {channel_id} (version {})", req.version);

        Ok(Response::new(LoadModelResponse {
            success: true,
            error: String::new(),
        }))
    }
}
