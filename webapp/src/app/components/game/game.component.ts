import { Component, OnInit } from '@angular/core';
import { GameBoardComponent } from '../game-board/game-board.component';
import { GameInfoComponent } from '../game-info/game-info.component';
import { GameService } from '../../services/game.service';
import { Board } from '../../types/board.model';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [GameBoardComponent, GameInfoComponent],
  templateUrl: './game.component.html',
  styleUrl: './game.component.scss',
})
export class GameComponent implements OnInit {
  board: Board | undefined;

  constructor(private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getNewGameBoard().subscribe((board: Board) => {
      this.board = board;
    });
  }

  applyMove(event: { gameId: string; moveIndex: number }): void {
    console.log('GameId', event.gameId);
    console.log('moveIndex', event.moveIndex);

    this.gameService
      .applyMove(event.gameId, event.moveIndex)
      .subscribe((board: Board) => {
        this.board = board;
      });
  }
}
