use alpha_cc_core::{Board, Move};

pub struct GameState<B: Board> {
    history: Vec<B>,
    moves: Vec<MoveRecord<B>>,
    initial_board: B,
}

#[allow(dead_code)]
pub struct MoveRecord<B: Board> {
    pub action_index: usize,
    pub mv: Move<B::Coord>,
}

impl<B: Board> GameState<B> {
    pub fn new(board: B) -> Self {
        let initial = board.clone();
        Self {
            history: vec![board],
            moves: Vec::new(),
            initial_board: initial,
        }
    }

    pub fn current_board(&self) -> &B {
        self.history.last().unwrap()
    }

    pub fn board_at(&self, index: usize) -> &B {
        &self.history[index]
    }

    pub fn apply_move(&mut self, action_index: usize) {
        let board = self.current_board();
        let moves = board.legal_moves();
        let mv = moves[action_index].clone();
        let new_board = board.apply_move(&mv);
        self.moves.push(MoveRecord { action_index, mv });
        self.history.push(new_board);
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    pub fn move_records(&self) -> &[MoveRecord<B>] {
        &self.moves
    }

    pub fn is_game_over(&self) -> bool {
        self.current_board().get_info().game_over
    }

    #[allow(dead_code)]
    pub fn winner(&self) -> i8 {
        self.current_board().get_info().winner
    }

    pub fn reset(&mut self) {
        self.history.truncate(1);
        self.history[0] = self.initial_board.clone();
        self.moves.clear();
    }
}
