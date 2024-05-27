import { Board } from './board.model';

export interface GameIO {
  gameId: string;
  boards: Board[];
}
