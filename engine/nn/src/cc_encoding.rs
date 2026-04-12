use std::collections::HashMap;

use alpha_cc_core::cc::{CCBoard, HexCoord, MAX_SIZE};
use alpha_cc_core::{Board, Coord, Move};

use crate::board_encoding::BoardEncoding;

impl BoardEncoding for CCBoard {
    const STATE_CHANNELS: usize = 2;
    const MOVE_BYTES: usize = 4;

    fn encode_state(&self, buf: &mut [f32]) {
        let (s, _) = self.get_sizes();
        let s = s as usize;
        for x in 0..s {
            for y in 0..s {
                let idx = x * s + y;
                let coord = HexCoord::new(x as u8, y as u8, s as u8);
                match self.get_content(&coord) {
                    1 => buf[idx] = 1.0,
                    2 => buf[s * s + idx] = 1.0,
                    _ => {}
                }
            }
        }
    }

    fn encode_move(mv: &Move<HexCoord>, buf: &mut [u8]) {
        let (fx, fy) = mv.from_coord.xy();
        let (tx, ty) = mv.to_coord.xy();
        buf[0] = fx;
        buf[1] = fy;
        buf[2] = tx;
        buf[3] = ty;
    }

    fn policy_size(board_size: usize) -> usize {
        board_size.pow(4)
    }

    fn policy_shape(board_size: usize) -> Vec<usize> {
        vec![board_size; 4]
    }

    fn move_to_policy_index(move_bytes: &[u8], board_size: usize) -> usize {
        let s = board_size;
        let fx = move_bytes[0] as usize;
        let fy = move_bytes[1] as usize;
        let tx = move_bytes[2] as usize;
        let ty = move_bytes[3] as usize;
        fx * s * s * s + fy * s * s + tx * s + ty
    }
}

// ── CC-specific policy helpers ──

/// Create a `[s, s, s, s]` boolean mask with `true` at legal move positions.
pub fn create_move_mask(moves: Vec<Move<HexCoord>>) -> [[[[bool; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE] {
    let mut mask = [[[[false; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE];
    for mv in moves {
        let from_x = mv.from_coord.x as usize;
        let from_y = mv.from_coord.y as usize;
        let to_x = mv.to_coord.x as usize;
        let to_y = mv.to_coord.y as usize;
        mask[from_x][from_y][to_x][to_y] = true;
    }
    mask
}

/// Map move indices to their (from, to) coordinate pairs.
pub fn create_move_index_map(moves: Vec<Move<HexCoord>>) -> HashMap<usize, (HexCoord, HexCoord)> {
    moves.iter().enumerate()
        .map(|(i, mv)| (i, (mv.from_coord, mv.to_coord)))
        .collect()
}
