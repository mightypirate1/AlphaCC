import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-game-controls',
  standalone: true,
  imports: [],
  templateUrl: './game-controls.component.html',
  styleUrl: './game-controls.component.scss',
})
export class GameControlsComponent {
  @Input() currentBoardIndex: number = 0;
  @Input() maxBoardIndex: number = 0;
  @Output() newBoardIndexEvent: EventEmitter<number> = new EventEmitter();

  changeCurrentMoveIndex(newCurrentBoardIndex: number): void {
    if (0 <= newCurrentBoardIndex && newCurrentBoardIndex <= this.maxBoardIndex)
      this.newBoardIndexEvent.emit(newCurrentBoardIndex);
  }
}
