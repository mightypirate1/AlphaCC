import { BoardIO } from './board-io.model';
import { Move } from './move.model';

export class Board {
  matrix: number[][];
  currentPlayer: number;
  gameOver: boolean;
  evaluation: number;
  winner: number;
  legalMoves: Move[];
  lastMove: Move;

  constructor(boardIo: BoardIO) {
    this.matrix = boardIo.matrix;
    this.currentPlayer = boardIo.currentPlayer;
    this.gameOver = boardIo.gameOver;
    this.evaluation = boardIo.evaluation;
    this.winner = boardIo.winner;
    this.legalMoves = boardIo.legalMoves;
    this.lastMove = boardIo.lastMove;
  }
}
