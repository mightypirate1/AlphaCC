use pyo3::prelude::*;
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use alpha_cc_core::Board;
use alpha_cc_core::cc::CCBoard;
use alpha_cc_nn::BoardEncoding;

use crate::core::PyBoard;
use super::nn_pred::PyNNPred;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn preds_from_logits<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    wdl_logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<PyBoard>,
    board_size: usize,
) -> PyResult<Vec<PyNNPred>> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let wdl_logits = wdl_logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("wdl_logits not contiguous: {e}")))?;
    let stride = CCBoard::policy_size(board_size);
    let mut preds = Vec::with_capacity(boards.len());

    for (i, py_board) in boards.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];
        let moves = py_board.0.legal_moves();

        let mut move_buf = [0u8; CCBoard::MOVE_BYTES];
        let move_logits: Vec<f32> = moves.iter().map(|m| {
            CCBoard::encode_move(m, &mut move_buf);
            let idx = CCBoard::move_to_policy_index(&move_buf, board_size);
            logits_slice[idx]
        }).collect();

        let pi = alpha_cc_nn::softmax(&move_logits);
        let wdl_row = &wdl_logits[i * 3..(i + 1) * 3];
        let wdl = alpha_cc_nn::softmax(wdl_row);
        preds.push(PyNNPred(alpha_cc_nn::NNPred::new(pi, [wdl[0], wdl[1], wdl[2]])));
    }

    Ok(preds)
}

type MoveCoords = Vec<(u8, u8, u8, u8)>;

#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn build_inference_request<'py>(
    py: Python<'py>,
    board: &PyBoard,
) -> (Bound<'py, numpy::PyArray3<f32>>, MoveCoords) {
    let (s, _) = board.0.get_sizes();
    let s = s as usize;
    let c = CCBoard::STATE_CHANNELS;

    let mut tensor_data = vec![0.0f32; c * s * s];
    board.0.encode_state(&mut tensor_data);

    let moves = board.0.legal_moves();
    let move_coords: MoveCoords = moves.iter().map(|m| {
        let mut buf = [0u8; CCBoard::MOVE_BYTES];
        CCBoard::encode_move(m, &mut buf);
        (buf[0], buf[1], buf[2], buf[3])
    }).collect();

    let arr = numpy::ndarray::Array3::<f32>::from_shape_vec([c, s, s], tensor_data).unwrap();
    let numpy_arr = numpy::IntoPyArray::into_pyarray(arr, py);
    (numpy_arr, move_coords)
}
