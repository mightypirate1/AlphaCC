import { Component } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatDialog } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { Observable, filter, map } from 'rxjs';

import { GameService } from '../../services/game.service';
import { GameInfoComponent } from '../game-info/game-info.component';
import { Game } from '../../types/game.model';

@Component({
  selector: 'app-game-controls',
  standalone: true,
  imports: [AsyncPipe, CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './game-controls.component.html',
  styleUrl: './game-controls.component.scss',
})
export class GameControlsComponent {
  currentBoardIndex$: Observable<number>;
  maxBoardIndex$: Observable<number>;

  constructor(private gameService: GameService, private dialog: MatDialog) {
    this.currentBoardIndex$ = gameService.currentBoardIndex();
    this.maxBoardIndex$ = gameService.game().pipe(
      filter((game): game is Game => game !== null),
      map((game) => {
        return game.boards.length - 1;
      })
    );
  }

  openDialog(): void {
    this.dialog.open(GameInfoComponent);
  }

  changeCurrentMoveIndex(newCurrentBoardIndex: number): void {
    this.gameService.changeCurrentMoveIndex(newCurrentBoardIndex);
  }
}
