import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CdkDragDrop, DragDropModule } from '@angular/cdk/drag-drop';

import { Board } from '../../types/board.model';
import { Point } from '../../types/point.model';
import { PegDirective } from '../../directives/peg.directive';

@Component({
  selector: 'app-game-board',
  standalone: true,
  templateUrl: './game-board.component.html',
  styleUrl: './game-board.component.scss',
  imports: [CommonModule, PegDirective, DragDropModule],
})
export class GameBoardComponent {
  @Input() board: Board | undefined;
  @Output() applyMoveEvent = new EventEmitter<{
    gameId: string;
    moveIndex: number;
  }>();
  colors = ['', 'orange', 'rebeccapurple'];
  selected: Point = { x: -1, y: -1 };

  constructor() {}

  setSelected(x: number, y: number): void {
    this.selected = { x: x, y: y };
  }

  drop(
    $event: CdkDragDrop<{ pegType: number; row: number; col: number }>
  ): void {
    const fromX = $event.item.dropContainer.data.row;
    const fromY = $event.item.dropContainer.data.col;
    const toX = $event.container.data.row;
    const toY = $event.container.data.col;
    if (this.board?.isLegalMove(fromX, fromY, toX, toY)) {
      const moveIndex = this.board?.getMoveIndex(fromX, fromY, toX, toY);
      this.applyMoveEvent.emit({
        gameId: this.board.gameId,
        moveIndex: moveIndex,
      });
    }
  }
}
