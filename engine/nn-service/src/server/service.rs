use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::proto::prediction_service_server::PredictionService;
use crate::proto::{PredictRequest, PredictResponse};
use crate::server::types::PendingPrediction;


pub struct NNService<F: Fn(u32) -> bool + Clone + Send + Sync + 'static> {
    pub route_map: Arc<HashMap<u32, mpsc::Sender<PendingPrediction>>>,
    pub model_id_ready: F,
}

#[tonic::async_trait]
impl<F: Fn(u32) -> bool + Clone + Send + Sync + 'static> PredictionService for NNService<F> {
    type PredictStream = ReceiverStream<Result<PredictResponse, Status>>;

    async fn predict(
        &self,
        request: Request<Streaming<PredictRequest>>,
    ) -> Result<Response<Self::PredictStream>, Status> {
        let mut inbound = request.into_inner();
        let route_map = self.route_map.clone();
        let model_id_ready = self.model_id_ready.clone();

        // Per-worker outbound channel.
        let (out_tx, out_rx) = mpsc::channel(64);

        // Bridge this worker's stream into the correct pipeline's batcher channel.
        tokio::spawn(async move {
            while let Some(result) = inbound.next().await {
                let req = match result {
                    Ok(r) => r,
                    Err(_) => break, // Stream error, worker probably disconnected.
                };

                // Hold request until the requested model is loaded.
                while !model_id_ready(req.model_id) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }

                // Route to the correct pipeline by model_id.
                let submit_tx = match route_map.get(&req.model_id) {
                    Some(tx) => tx,
                    None => {
                        // Unknown model_id — fall back to first pipeline.
                        match route_map.values().next() {
                            Some(tx) => tx,
                            None => break,
                        }
                    }
                };

                let (reply_tx, reply_rx) = oneshot::channel();

                if submit_tx
                    .send(PendingPrediction::new(req, reply_tx))
                    .await
                    .is_err()
                {
                    break; // Batcher gone.
                }

                match reply_rx.await {
                    Ok(resp) => {
                        if out_tx.send(Ok(resp)).await.is_err() {
                            break; // Worker disconnected.
                        }
                    }
                    Err(_) => break, // Batcher dropped our reply channel.
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(out_rx)))
    }
}
