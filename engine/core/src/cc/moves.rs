use std::collections::HashSet;
use indexmap::IndexSet;
use crate::Move;
use crate::cc::{CCBoard, HexCoord};


pub fn find_all_moves(board: &CCBoard) -> Vec<Move<HexCoord>> {
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
    let mut moves: Vec<Move<HexCoord>> = Vec::new();
    let mut from_coord: HexCoord;
    let mut from_coords: IndexSet<HexCoord> = IndexSet::new();

    for x in 0..size {
        for y in 0..size {
            from_coord = HexCoord::new(x, y , board.get_size());
            if board.coord_is_player1(&from_coord) {
                from_coords.insert(from_coord);
                for to_coord in from_coord.get_all_neighbours_arr(1).into_iter().flatten() {
                    if board.coord_is_empty(&to_coord)
                    || board.coord_is_player2_home(&to_coord)
                    && board.coord_is_player2(&to_coord) {
                        moves.push(
                            Move {
                                from_coord,
                                to_coord,
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


fn find_all_jump_moves (board: &CCBoard, coord: &HexCoord) -> Vec<Move<HexCoord>> {
    let mut jump_moves: Vec<Move<HexCoord>> = Vec::new();
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
    board: &CCBoard,
    starting_coord: &HexCoord,
    current_coord: &HexCoord,
    final_positions: &'a mut HashSet<HexCoord>,
    jump_moves: &'a mut Vec<Move<HexCoord>>,
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

    for direction in 0..6 {
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
