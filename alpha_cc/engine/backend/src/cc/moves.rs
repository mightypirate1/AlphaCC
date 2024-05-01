extern crate pyo3;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use crate::cc::{Board, HexCoord, Move};
use crate::cc::board::MAX_SIZE;


#[pyfunction]
pub fn create_move_mask(moves: Vec<Move>) -> [[[[bool; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE] {
    /*
    notice that this mask needs to be cropped to match the board size
     */

    let mut mask = [[[[false; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE];
    for r#move in moves {
        mask[r#move.from_coord.x][r#move.from_coord.y][r#move.to_coord.x][r#move.to_coord.y] = true;
    }
    mask
}
#[pyfunction]
pub fn create_move_index_map(moves: Vec<Move>) -> HashMap<usize, (HexCoord, HexCoord)> {
    let mut move_index_map:HashMap<usize, (HexCoord, HexCoord)> = HashMap::new();
    for (i, r#move) in moves.iter().enumerate() {
        move_index_map.insert(i, (r#move.from_coord, r#move.to_coord));
    }
    move_index_map
}


pub fn find_all_moves(board: &Board) -> Vec<Move> {
    let size = board.get_size();
    let mut moves: Vec<Move> = Vec::new();
    let mut coord: HexCoord;
    let mut from_coords: HashSet<HexCoord> = HashSet::new();
    
    for x in 0..size {
        for y in 0..size {
            if board.get_matrix()[x][y] == 1 {
                coord = HexCoord::create(x, y , board.get_size());
                for direction in coord.get_all_directions(){
                    if let Some(to_coord) = coord.get_neighbor(direction, 1) {
                        if board.coord_is_empty(&to_coord) {
                            moves.push(
                                Move {
                                    from_coord: coord,
                                    to_coord,
                                    path: Vec::new(),
                                }
                            );
                            from_coords.insert(coord);
                        }
                    }
                }
            }
        }
    }
    for coord in from_coords {
        moves.extend(find_all_jump_moves(board, &coord));
    }
    moves
}


fn find_all_jump_moves (board: &Board, coord: &HexCoord) -> Vec<Move> {
    let mut jump_moves: Vec<Move> = Vec::new();
    let mut final_positions: HashSet<HexCoord> = HashSet::new();
    final_positions.insert(*coord);
    _recusive_exporation(
        board,
        coord, 
        coord,
        &mut final_positions,
        &mut jump_moves,
    );
    jump_moves
}


fn _recusive_exporation<'a>(
    board: &Board,
    starting_coord: &HexCoord,
    current_coord: &HexCoord,
    final_positions: &'a mut HashSet<HexCoord>,
    jump_moves: &'a mut Vec<Move>,
) { 
    /*
    does not explicitly return a value; instead it adds them to the `jump_moves` vec
     */
    for direction in current_coord.get_all_directions() {
        let mb_target_coord = current_coord.get_neighbor(direction, 2);
        let mb_intermediate_coord = current_coord.get_neighbor(direction, 1);
        if mb_target_coord.is_some() && mb_intermediate_coord.is_some() {
            let target_coord = mb_target_coord.unwrap();
            let intermediate_coord = mb_intermediate_coord.unwrap();
            if board.coord_is_empty(&target_coord)
                && !board.coord_is_empty(&intermediate_coord)
                && !final_positions.contains(&target_coord) {
                    // If `current_coord` -> `target_coord` woul be a legal jump,
                    // and this is the fist time we encounter `target_coord`:
                    // - Remember that we already know we can get to `target_coord` from `starting_coord`.
                    // - Add `starting_coord` -> `target_coord` to list of Jumps.
                    final_positions.insert(target_coord);
                    jump_moves.push(
                        Move {
                            from_coord: *starting_coord,
                            to_coord: target_coord,
                            path: Vec::new(),
                        }
                    );
                    _recusive_exporation(
                        board,
                        starting_coord,
                        &target_coord,
                        final_positions,
                        jump_moves,
                    );
                }
        }
    }
}
