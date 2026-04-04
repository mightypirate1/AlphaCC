use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;
use crate::cc::Move;

pub struct GameState {
    history: Vec<Board>,
    moves: Vec<MoveRecord>,
    board_size: u8,
}

pub struct MoveRecord {
    pub action_index: usize,
    pub mv: Move,
}

impl GameState {
    pub fn new(board_size: u8) -> Self {
        Self {
            history: vec![Board::create(board_size as usize)],
            moves: Vec::new(),
            board_size,
        }
    }

    pub fn current_board(&self) -> &Board {
        self.history.last().unwrap()
    }

    pub fn board_at(&self, index: usize) -> &Board {
        &self.history[index]
    }

    pub fn apply_move(&mut self, action_index: usize) {
        let board = self.current_board();
        let moves = find_all_moves(board);
        let mv = moves[action_index].clone();
        let new_board = board.apply(&mv);
        self.moves.push(MoveRecord { action_index, mv });
        self.history.push(new_board);
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    pub fn move_records(&self) -> &[MoveRecord] {
        &self.moves
    }

    pub fn is_game_over(&self) -> bool {
        self.current_board().get_info().game_over
    }

    pub fn winner(&self) -> i8 {
        self.current_board().get_info().winner
    }

    pub fn board_size(&self) -> u8 {
        self.board_size
    }

    pub fn reset(&mut self) {
        self.history.truncate(1);
        self.history[0] = Board::create(self.board_size as usize);
        self.moves.clear();
    }
}
