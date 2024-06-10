import { Component } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { Observable, combineLatest, map } from 'rxjs';

import { GameService } from '../../services/game.service';

@Component({
  selector: 'app-game-info',
  standalone: true,
  imports: [AsyncPipe],
  templateUrl: './game-info.component.html',
  styleUrl: './game-info.component.scss',
})
export class GameInfoComponent {
  gameId$: Observable<string>;
  currentPlayer$: Observable<number>;
  gameOver$: Observable<boolean>;
  winner$: Observable<number>;

  constructor(private gameService: GameService) {
    this.gameId$ = gameService.game().pipe(map((game) => game.gameId));
    this.currentPlayer$ = combineLatest([
      gameService.game(),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].currentPlayer));
    this.gameOver$ = combineLatest([
      gameService.game(),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].gameOver));
    this.winner$ = combineLatest([
      gameService.game(),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].winner));
  }
}
