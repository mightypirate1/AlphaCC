import { Component } from '@angular/core';

import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameControlsComponent } from '../game-controls/game-controls.component';

@Component({
  selector: 'app-game',
  standalone: true,
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
  imports: [GameBoardComponent, GameInfoComponent, GameControlsComponent],
})
export class GameComponent {}
