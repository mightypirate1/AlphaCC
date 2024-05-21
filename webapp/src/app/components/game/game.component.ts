import { Component, OnInit } from '@angular/core';
import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameService } from '../../services/game.service';
import { Game } from '../../types/game.model';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [GameBoardComponent, GameInfoComponent],
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
})
export class GameComponent implements OnInit {
  game: Game | null = null;
  currentBoardIndex = 0;

  constructor(private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getNewGameBoard().subscribe((game: Game) => {
      this.game = game;
    });
  }

  applyMove(event: number): void {
    if (this.game === null) return;
    this.gameService
      .applyMove(this.game.gameId, event)
      .subscribe((game: Game) => {
        this.game = game;
        this.currentBoardIndex++;
      });
  }
}
