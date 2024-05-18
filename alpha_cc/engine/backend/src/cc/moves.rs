extern crate pyo3;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use indexmap::IndexSet;
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
    /*
    there are 2 types of moves (both with 2 variants):
    - jump moves (see `find_all_jump_moves`)
    - walk moves (moving to an adjacent cell)

    both of these require that the target cell is empty, - except:
    if the taget cell is in the opponent's home, and is occupied by one of the
    opponent's stones; then the move is still legal, and the opponent's stone
    will be swap places with the moved stone.
     */
    let size = board.get_size();
    let mut moves: Vec<Move> = Vec::new();
    let mut from_coord: HexCoord;
    let mut from_coords: IndexSet<HexCoord> = IndexSet::new();
    
    for x in 0..size {
        for y in 0..size {
            if board.get_matrix()[x][y] == 1 {
                from_coord = HexCoord::create(x, y , board.get_size());
                from_coords.insert(from_coord);
                for to_coord in from_coord.get_all_neighbours(1) {
                    if board.coord_is_empty(&to_coord)
                    || board.coord_is_player2_home(&to_coord)
                    && board.coord_is_player2(&to_coord) {
                        moves.push(
                            Move {
                                from_coord,
                                to_coord,
                                path: Vec::new(),
                            }
                        );
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
    there are 2 types of jumps:
    1. standard jump:
        - if a stone is next to another one, and the next cell (in the same direction)
            is empty, the stone can jump over the other stone to the empty cell.
        - if from there, there is another jump available, the stone can continue jumping.
        - one can chain jumps together like this until there are no more jumps available,
            or stop anywhere along the way.
    2. swap jump:
        - if a potential jump is blocked by an opponent's stone that is in it's home still,
            the stone can swap places with the opponent's stone. i.e. it is not possible to
            block a jump by camping in one's home.
        - this type of jump does not allow for further jumps.

    NOTE: this function does not explicitly return a value; instead it adds them to the 
        `jump_moves` vec.
     */

    for direction in current_coord.get_all_directions() {
        if let Some(target_coord) = current_coord.get_neighbor(direction, 2) {
            if let Some(intermediate_coord) = current_coord.get_neighbor(direction, 1) {
            let is_standard_jump = board.coord_is_empty(&target_coord)
                && !board.coord_is_empty(&intermediate_coord)
                && !final_positions.contains(&target_coord);
            
            let is_swap_jump = board.coord_is_player2(&target_coord)
                && board.coord_is_player2_home(&target_coord)
                && !board.coord_is_empty(&intermediate_coord)
                && !final_positions.contains(&target_coord);

            if is_standard_jump || is_swap_jump {
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
                }
            if is_standard_jump {
                // If this is a standard jump, we can continue exploring from `target_coord`.
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
}
