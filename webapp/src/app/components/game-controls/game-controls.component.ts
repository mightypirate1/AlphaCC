import { Component } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { Observable, map } from 'rxjs';

import { GameService } from '../../services/game.service';

@Component({
  selector: 'app-game-controls',
  standalone: true,
  imports: [AsyncPipe, CommonModule],
  templateUrl: './game-controls.component.html',
  styleUrl: './game-controls.component.scss',
})
export class GameControlsComponent {
  currentBoardIndex$: Observable<number>;
  maxBoardIndex$: Observable<number>;

  constructor(private gameService: GameService) {
    this.currentBoardIndex$ = this.gameService.currentBoardIndex();
    this.maxBoardIndex$ = this.gameService
      .game()
      .pipe(map((game) => game.boards.length - 1));
  }

  changeCurrentMoveIndex(newCurrentBoardIndex: number): void {
    this.gameService.changeCurrentMoveIndex(newCurrentBoardIndex);
  }
}
