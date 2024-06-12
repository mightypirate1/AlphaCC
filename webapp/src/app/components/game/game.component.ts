import { Component } from '@angular/core';

import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameControlsComponent } from '../game-controls/game-controls.component';
import { GameService } from '../../services/game.service';
import { ActivatedRoute, Router } from '@angular/router';
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
        const player = +url[0].parameters['player'];
        this.gameService.setActiveGame(gameId);
        if (player >= 0) {
          this.gameService.setActivePlayer(gameId, player);
          delete url[0].parameters['player'];
          this.router.navigate([]);
        }
      })
      .unsubscribe();
  }
}
