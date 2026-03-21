use pyo3::prelude::*;

use crate::cc::predictions::nn_pred::NNPred;
use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;

#[pyfunction]
pub fn preds_from_logits<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<Board>,
    board_size: usize,
) -> PyResult<Vec<NNPred>> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let values = values_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values not contiguous: {e}")))?;
    let s = board_size;
    let stride = s * s * s * s;
    let mut preds = Vec::with_capacity(boards.len());

    for (i, board) in boards.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];
        let moves = find_all_moves(board);

        let move_logits: Vec<f32> = moves.iter().map(|m| {
            let fx = m.from_coord.x as usize;
            let fy = m.from_coord.y as usize;
            let tx = m.to_coord.x as usize;
            let ty = m.to_coord.y as usize;
            logits_slice[fx * s * s * s + fy * s * s + tx * s + ty]
        }).collect();

        let pi = softmax(&move_logits);
        let value = values[i];
        preds.push(NNPred::new(pi, value));
    }

    Ok(preds)
}

type MoveCoords = Vec<(u8, u8, u8, u8)>;

/// Expose board encoding for testing.
/// Returns (tensor_data as numpy (2, s, s), move_coords).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn build_inference_request<'py>(
    py: Python<'py>,
    board: &Board,
) -> (Bound<'py, numpy::PyArray3<f32>>, MoveCoords) {
    let s = board.get_size() as usize;
    let matrix = board.get_matrix();
    let mut tensor_data = vec![0.0f32; 2 * s * s];
    for (x, row) in matrix.iter().enumerate().take(s) {
        for (y, &val) in row.iter().enumerate().take(s) {
            let idx = x * s + y;
            if val == 1 {
                tensor_data[idx] = 1.0;
            } else if val == 2 {
                tensor_data[s * s + idx] = 1.0;
            }
        }
    }

    let moves = find_all_moves(board);
    let move_coords: MoveCoords = moves.iter().map(|m| {
        (m.from_coord.x, m.from_coord.y, m.to_coord.x, m.to_coord.y)
    }).collect();

    let arr = numpy::ndarray::Array3::<f32>::from_shape_vec([2, s, s], tensor_data).unwrap();
    let numpy_arr = numpy::IntoPyArray::into_pyarray(arr, py);
    (numpy_arr, move_coords)
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
