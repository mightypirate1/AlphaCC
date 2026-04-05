use pyo3::prelude::*;
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use alpha_cc_core::moves::find_all_moves;

use crate::core::PyBoard;
use super::nn_pred::PyNNPred;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn preds_from_logits<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<PyBoard>,
    board_size: usize,
) -> PyResult<Vec<PyNNPred>> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let values = values_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values not contiguous: {e}")))?;
    let s = board_size;
    let stride = s * s * s * s;
    let mut preds = Vec::with_capacity(boards.len());

    for (i, py_board) in boards.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];
        let moves = find_all_moves(&py_board.0);

        let move_logits: Vec<f32> = moves.iter().map(|m| {
            let fx = m.from_coord.x as usize;
            let fy = m.from_coord.y as usize;
            let tx = m.to_coord.x as usize;
            let ty = m.to_coord.y as usize;
            logits_slice[fx * s * s * s + fy * s * s + tx * s + ty]
        }).collect();

        let pi = alpha_cc_nn::softmax(&move_logits);
        let value = values[i];
        preds.push(PyNNPred(alpha_cc_nn::NNPred::new(pi, value)));
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
    let s = board.0.get_size() as usize;
    let matrix = board.0.get_matrix();
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

    let moves = find_all_moves(&board.0);
    let move_coords: MoveCoords = moves.iter().map(|m| {
        (m.from_coord.x, m.from_coord.y, m.to_coord.x, m.to_coord.y)
    }).collect();

    let arr = numpy::ndarray::Array3::<f32>::from_shape_vec([2, s, s], tensor_data).unwrap();
    let numpy_arr = numpy::IntoPyArray::into_pyarray(arr, py);
    (numpy_arr, move_coords)
}
