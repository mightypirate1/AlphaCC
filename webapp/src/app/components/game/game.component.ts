import { Component } from '@angular/core';

import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameControlsComponent } from '../game-controls/game-controls.component';
import { GameInitComponent } from '../game-init/game-init.component';
import { NewGameFormData } from '../../types/new-game-form-data';
import { GameService } from '../../services/game.service';

@Component({
  selector: 'app-game',
  standalone: true,
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
  imports: [
    GameBoardComponent,
    GameInfoComponent,
    GameControlsComponent,
    GameInitComponent,
  ],
})
export class GameComponent {
  gameActive: boolean = false;

  constructor(private gameService: GameService) {}

  public formGroupChange(formData: NewGameFormData) {
    this.gameService.newGame(formData.gameId, formData.gameSize);
    this.gameActive = true;
  }
}
