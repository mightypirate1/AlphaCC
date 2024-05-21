import { Component, OnInit } from '@angular/core';
import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameService } from '../../services/game.service';
import { Game } from '../../types/game.model';
import { GameControlsComponent } from '../game-controls/game-controls.component';

@Component({
  selector: 'app-game',
  standalone: true,
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
  imports: [GameBoardComponent, GameInfoComponent, GameControlsComponent],
})
export class GameComponent implements OnInit {
  game: Game | undefined;
  currentBoardIndex: number = 0;

  constructor(private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getNewGameBoard().subscribe((game: Game) => {
      this.game = game;
    });
  }

  applyMove(moveIndex: number): void {
    if (this.game === undefined) {
      throw Error('Trying to apply a move on undefined game object.');
    }
    this.gameService
      .applyMove(this.game.gameId, moveIndex)
      .subscribe((game: Game) => {
        this.game = game;
        this.currentBoardIndex++;
      });
  }

  changeCurrentBoardIndex(newIndex: number): void {
    this.currentBoardIndex = newIndex;
  }
}
