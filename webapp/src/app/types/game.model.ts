import { Board } from './board.model';
import { GameIO } from './game-io.model';

export class Game {
  gameId: string;
  boards: Board[];

  constructor(gameIo: GameIO) {
    this.gameId = gameIo.gameId;
    this.boards = gameIo.boards;
  }
}
