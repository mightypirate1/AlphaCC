import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-game-info',
  standalone: true,
  imports: [],
  templateUrl: './game-info.component.html',
  styleUrl: './game-info.component.scss',
})
export class GameInfoComponent {
  @Input() gameId: string = '';
  @Input() currentPlayer: number = -1;
  @Input() gameOver: boolean = false;
  @Input() winner: number = -1;
}
