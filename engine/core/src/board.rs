/*
Board is designed such that under the hood; whoever is
the current player, their pieces are always ones 1,
and the opponent's are 2.

This may lead to some confusion, but I think its faster
and less bug prone

*/

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::{BoardInfo, HexCoord, Move, WDL};
use crate::moves::find_all_moves;
use crate::dtypes;

pub const MAX_SIZE: usize = 9;

pub type BoardMatrix = [[dtypes::BoardContent; MAX_SIZE]; MAX_SIZE];


#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct Board {
    size: dtypes::BoardSize,
    duration: dtypes::GameDuration,
    home_size: dtypes::BoardSize,
    home_capacity: dtypes::HomeCapacity,
    matrix: BoardMatrix,
    current_player: i8,
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl Eq for Board {}

impl Hash for Board {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.matrix.hash(state);
    }
}

impl Board {

    pub fn create(size: usize) -> Board {
        if ![3, 5, 7, 9].contains(&size) {
            panic!("supports sizes: 3, 5, 7, 9");
        }
        let size = size as dtypes::BoardSize;
        Board {
            size,
            duration: 0,
            home_size: Board::home_size(size),
            home_capacity: Board::home_capacity(size),
            matrix: Board::initialize_matrix(size),
            current_player: 1,
        }
    }

    pub fn get_size(&self) -> dtypes::BoardSize {
        self.size
    }

    pub fn get_content(&self, coord: &HexCoord) -> i8 {
        self.matrix[coord.x as usize][coord.y as usize]
    }

    pub fn coord_is_empty(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 0
    }

    pub fn coord_is_player1(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 1
    }

    pub fn coord_is_player2(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 2
    }

    pub fn coord_is_player1_home(&self, coord: &HexCoord) -> bool {
        Board::xy_start_val(coord.x, coord.y, self.size) == 1
    }

    pub fn coord_is_player2_home(&self, coord: &HexCoord) -> bool {
        Board::xy_start_val(coord.x, coord.y, self.size) == 2
    }

    pub fn apply_move(&self, r#move: &Move) -> Board {
        /*
        we attempt to save a copy by flipping once (copy),
        and then perform the move on the flipped matrix:
        - flipped board
        - flipped coords
        - played piece is 2 (flipped)

        most moves will leave the from_coord empty, but some moves
        swap the stone with the opponents stone. thus we will put
        whatever content is at the to_coord cell at the from_coord cell,
        and trust the move finding algorithm to have checked that the
        result is valid.
        */

        let mut matrix = self.flipped_matrix();
        let flipped_from_coord = r#move.from_coord.flip();
        let flipped_to_coord = r#move.to_coord.flip();
        let flipped_to_x = flipped_to_coord.x as usize;
        let flipped_to_y = flipped_to_coord.y as usize;
        let flipped_from_x = flipped_from_coord.x as usize;
        let flipped_from_y = flipped_from_coord.y as usize;
        let to_coord_content = matrix[flipped_to_x][flipped_to_y];

        matrix[flipped_from_x][flipped_from_y] = to_coord_content;
        matrix[flipped_to_x][flipped_to_y] = 2;  // 2 is the current player (since the board is flipped)
        Board {
            size: self.size,
            duration: self.duration + 1,
            home_size: self.home_size,
            home_capacity: self.home_capacity,
            matrix,
            current_player: if self.current_player == 1 {2} else {1},
        }
    }

    pub fn compute_hash(&self) -> dtypes::BoardHash {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn compute_wdl_and_winner(&self) -> (WDL, i8) {
        /*
        Compute WDL (win/draw/loss) and winner for a board state.

        For terminal states:
            Winner determined, WDL is one-hot from current player's perspective.

        For non-terminal states:
            P(win) and P(loss) scale with how many pieces each player has landed
            in the goal. P(draw) scales with the unresolved portion.
         */
        let s = self.size;
        let hs = self.home_size;
        let cap = self.home_capacity as f32;

        let mut n_p1_stones_in_p2_home: i32 = 0;
        let mut goal_is_full: bool = true;
        for x in (s-hs)..s {
            for y in (s-hs)..s {
                if Board::xy_start_val(x, y, self.size) == 2 {
                    let coord = HexCoord::new(x, y, self.size);
                    let content = self.get_content(&coord);
                    if content == 0 {
                        goal_is_full = false;
                    }
                    if content == 1 {
                        n_p1_stones_in_p2_home += 1;
                    }
                }
            }
        }
        // p1 wins (current player)
        if goal_is_full && n_p1_stones_in_p2_home > 0 {
            return (WDL::win(), self.current_player);
        }

        let mut n_p2_stones_in_p1_home: i32 = 0;
        goal_is_full = true;
        for x in 0..hs {
            for y in 0..hs {
                if Board::xy_start_val(x, y, self.size) == 1 {
                    let coord = HexCoord::new(x, y, self.size);
                    let content = self.get_content(&coord);
                    if content == 0 {
                        goal_is_full = false;
                    }
                    if content == 2 {
                        n_p2_stones_in_p1_home += 1;
                    }
                }
            }
        }
        // p2 wins (opponent)
        if goal_is_full && n_p2_stones_in_p1_home > 0 {
            return (WDL::loss(), 3 - self.current_player);
        }

        // Non-terminal: heuristic WDL from current player's perspective.
        // Draw is high when the game is early (low avg) and close (low gap).
        // Win/loss split the decisive portion proportionally to progress.
        let my_progress = n_p1_stones_in_p2_home as f32 / cap;
        let their_progress = n_p2_stones_in_p1_home as f32 / cap;
        let avg = (my_progress + their_progress) / 2.0;
        let gap = (my_progress - their_progress).abs();
        let draw = (1.0 - avg) * (1.0 - gap);
        let decisive = my_progress + their_progress;
        let wdl = if decisive > 0.0 {
            WDL {
                win: (1.0 - draw) * my_progress / decisive,
                draw,
                loss: (1.0 - draw) * their_progress / decisive,
            }
        } else {
            WDL { win: 0.0, draw: 1.0, loss: 0.0 }
        };
        (wdl, 0)
    }

    #[allow(clippy::needless_range_loop)]
    fn flipped_matrix(&self) -> BoardMatrix {
        /*
        Used when playing moves to make it look to the next player like they are the main character
         */
        let mut matrix = Board::empty_matrix();
        // manual version for now

        for x in 0..self.size {
            for y in 0..self.size {
                let coord = HexCoord::new(x, y, self.size);
                let flipped_content = self.get_content(&coord.flip());
                matrix[x as usize][y as usize] = match flipped_content {
                    0 => 0,
                    1 => 2,
                    2 => 1,
                    _ => {
                        unreachable!("invalid value on board: {flipped_content}")
                    }
                };
            }
        }
        matrix
    }

    fn empty_matrix() -> BoardMatrix {
        [[0; MAX_SIZE]; MAX_SIZE]
    }

    #[allow(clippy::needless_range_loop)]
    fn initialize_matrix(size: dtypes::BoardSize) -> BoardMatrix {
        let mut matrix = Board::empty_matrix();
        for x in 0..MAX_SIZE.try_into().unwrap() {
            for y in 0..MAX_SIZE.try_into().unwrap() {
                matrix[x as usize][y as usize] = Board::xy_start_val(x, y, size);
            }
        }
        matrix
    }

    fn xy_start_val(x: dtypes::BoardSize, y: dtypes::BoardSize, size: dtypes::BoardSize) -> i8 {
        // player 1 home
        let home_size = Board::home_size(size);
        if x + y < home_size {
            return 1;
        }
        // player 2 home
        if x + y >= home_size + size && x < size && y < size {
            return 2;
        }
        0
    }

    fn home_size(size: dtypes::BoardSize) -> dtypes::BoardSize {
        (size - 1) / 2
    }

    fn home_capacity(size: dtypes::BoardSize) -> dtypes::HomeCapacity {
        let hs = Board::home_size(size) as usize;
        ((hs * (hs + 1)) / 2) as dtypes::HomeCapacity
    }

    pub fn serialize_rs(&self) -> dtypes::EncBoard {
        bitcode::encode(self)
    }

    pub fn deserialize_rs(data: &[u8]) -> Board {
        bitcode::decode(data)
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize board state: {}", e)
            })
    }
}


impl Board {
    pub fn reset(&self) -> Board {
        Board::create(self.size as usize)
    }

    pub fn get_moves(&self) -> Vec<Move> {
        find_all_moves(self)
    }

    pub fn get_next_states(&self) -> Vec<Board> {
        let mut next_states: Vec<Board> = Vec::new();
        for r#move in self.get_moves() {
            next_states.push(self.apply_move(&r#move));
        }
        next_states
    }

    pub fn get_matrix(&self) -> BoardMatrix {
        self.matrix
    }

    pub fn apply(&self, r#move: &Move) -> Board {
        self.apply_move(r#move)
    }

    pub fn get_unflipped_matrix(&self) -> BoardMatrix {
        /*
        unflips the matrix if needed to make it intelligable for humans
         */
        if self.current_player == 1 {
            return self.matrix
        }
        self.flipped_matrix()
    }

    // Render colors (truecolor ANSI: R, G, B)
    const COLOR_EMPTY: (u8, u8, u8) = (150, 100, 200); // purple
    const COLOR_P1: (u8, u8, u8) = (255, 140, 50);     // orange
    const COLOR_P2: (u8, u8, u8) = (255, 105, 180);    // pink

    pub fn render(&self) {
        let matrix = self.get_unflipped_matrix();
        println!();
        for (i, row) in matrix[0..self.size as usize].iter().enumerate() {
            for _ in 0..i {
                print!(" ");
            }
            for val in row[0..self.size as usize].iter() {
                let (r, g, b, ch) = match val {
                    1 => (Self::COLOR_P1.0, Self::COLOR_P1.1, Self::COLOR_P1.2, "⬢"),
                    2 => (Self::COLOR_P2.0, Self::COLOR_P2.1, Self::COLOR_P2.2, "⬢"),
                    _ => (Self::COLOR_EMPTY.0, Self::COLOR_EMPTY.1, Self::COLOR_EMPTY.2, "⬡"),
                };
                print!("\x1b[38;2;{r};{g};{b}m{ch}\x1b[0m ");
            }
            println!();
        }
        println!();
        let (cr, cg, cb) = match self.current_player {
            1 => Self::COLOR_P1,
            _ => Self::COLOR_P2,
        };
        println!("current player: \x1b[38;2;{cr};{cg};{cb}m⬢\x1b[0m ({})", self.current_player);
    }

}

impl Board {
    pub fn get_info(&self) -> BoardInfo {
        let (wdl, winner) = self.compute_wdl_and_winner();
        BoardInfo {
            current_player: self.current_player,
            winner,
            wdl,
            size: self.size,
            duration: self.duration,
            game_over: winner > 0,
        }
    }
}


