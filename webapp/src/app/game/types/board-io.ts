import { Move } from './move.model';

export interface BoardIO {
  gameId: string;
  matrix: number[][];
  currentPlayer: number;
  gameOver: boolean;
  winner: number;
  legalMoves: Move[];
  lastMove: Move;
}
