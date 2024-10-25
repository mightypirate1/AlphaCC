import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameControlsComponent } from '../game-controls/game-controls.component';
import { GameService } from '../../services/game.service';
import { GamePlotsComponent } from '../game-plots/game-plots.component';

@Component({
  selector: 'app-game',
  standalone: true,
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
  imports: [
    GameBoardComponent,
    GameInfoComponent,
    GameControlsComponent,
    GamePlotsComponent,
  ],
})
export class GameComponent {
  constructor(
    private gameService: GameService,
    private activatedRoute: ActivatedRoute,
    private router: Router
  ) {
    this.activatedRoute.url
      .subscribe((url) => {
        const gameId = url[0].path;
        this.gameService.setActiveGame(gameId);
        try {
          const playersArray = url[0].parameters['players'].split(',');
          this.gameService.setPlayersSettings(playersArray);
          delete url[0].parameters['players'];
        } catch (TypeError) {
          this.gameService.setPlayersSettings(['HUMAN', 'HUMAN']);
        }
        this.router.navigate([]);
      })
      .unsubscribe();
  }
}
