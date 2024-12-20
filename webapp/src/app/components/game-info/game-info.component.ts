import { Component } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule } from '@angular/material/dialog';
import { Observable, combineLatest, filter, map } from 'rxjs';

import { GameService } from '../../services/game.service';
import { Game } from '../../types/game.model';

@Component({
  selector: 'app-game-info',
  standalone: true,
  imports: [AsyncPipe, MatButtonModule, MatDialogModule],
  templateUrl: './game-info.component.html',
  styleUrl: './game-info.component.scss',
})
export class GameInfoComponent {
  gameId$: Observable<string>;
  currentPlayer$: Observable<number>;
  gameOver$: Observable<boolean>;
  winner$: Observable<number>;

  constructor(private gameService: GameService) {
    this.gameId$ = gameService.game().pipe(
      filter((game): game is Game => game !== null),
      map((game) => game.gameId)
    );
    this.currentPlayer$ = combineLatest([
      gameService.game().pipe(filter((game): game is Game => game !== null)),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].currentPlayer));
    this.gameOver$ = combineLatest([
      gameService.game().pipe(filter((game): game is Game => game !== null)),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].gameOver));
    this.winner$ = combineLatest([
      gameService.game().pipe(filter((game): game is Game => game !== null)),
      gameService.currentBoardIndex(),
    ]).pipe(map(([game, boardIndex]) => game.boards[boardIndex].winner));
  }
}
