import { Component } from '@angular/core';

import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameControlsComponent } from '../game-controls/game-controls.component';
import { GameService } from '../../services/game.service';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-game',
  standalone: true,
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
  imports: [GameBoardComponent, GameInfoComponent, GameControlsComponent],
})
export class GameComponent {
  constructor(
    private gameService: GameService,
    private activatedRoute: ActivatedRoute,
    private router: Router
  ) {
    this.activatedRoute.url
      .subscribe((url) => {
        this.gameService.setActiveGame(url[0].path);
        delete url[0].parameters['size'];
        this.router.navigate([]);
      })
      .unsubscribe();
  }
}
