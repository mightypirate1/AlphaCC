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

  getMoveIndex(fromX: number, fromY: number, toX: number, toY: number): number {
    const matchingMove = this.legalMoves.find((move) => {
      return (
        move.fromCoord.x === fromX &&
        move.fromCoord.y === fromY &&
        move.toCoord.x === toX &&
        move.toCoord.y === toY
      );
    });
    return matchingMove ? matchingMove.index : -1;
  }

  isLegalMove(fromX: number, fromY: number, toX: number, toY: number): boolean {
    return this.legalMoves.some((move) => {
      return (
        move.fromCoord.x === fromX &&
        move.fromCoord.y === fromY &&
        move.toCoord.x === toX &&
        move.toCoord.y === toY
      );
    });
  }

  isCoordOfLastMove(x: number, y: number): boolean {
    if (this.lastMove === null) return false;
    return (
      (this.lastMove.fromCoord.x === x && this.lastMove.fromCoord.y === y) ||
      (this.lastMove.toCoord.x === x && this.lastMove.toCoord.y === y)
    );
  }

  isLegalMoveSource(fromX: number, fromY: number): boolean {
    return this.legalMoves
      .map((move) => move.fromCoord)
      .some((pt) => {
        return pt.x === fromX && pt.y === fromY;
      });
  }
}
