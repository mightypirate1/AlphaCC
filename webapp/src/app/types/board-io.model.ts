import { Move } from './move.model';

export interface BoardIO {
  matrix: number[][];
  currentPlayer: number;
  gameOver: boolean;
  evaluation: number;
  winner: number;
  legalMoves: Move[];
  lastMove: Move;
}
