import { Board } from './board.model';
import { GameIO } from './game-io';

export class Game {
  gameId: string;
  boards: Board[];

  constructor(gameIo: GameIO) {
    this.gameId = gameIo.gameId;
    this.boards = [];

    gameIo.boards.forEach((boardIo) => {
      this.boards.push(new Board(boardIo));
    });
  }
}
