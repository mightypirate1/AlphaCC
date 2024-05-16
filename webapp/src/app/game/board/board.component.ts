import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CdkDragDrop, DragDropModule } from '@angular/cdk/drag-drop';

import { Board } from '../types/board.model';
import { Point } from '../types/point.model';
import { PegDirective } from '../directives/peg.directive';
import { BoardService } from '../services/board.service';

@Component({
  selector: 'app-board',
  standalone: true,
  templateUrl: './board.component.html',
  styleUrl: './board.component.scss',
  imports: [CommonModule, PegDirective, DragDropModule],
})
export class BoardComponent implements OnInit {
  board: Board | undefined;
  colors = ['', 'orange', 'rebeccapurple'];
  selected: Point = { x: -1, y: -1 };

  constructor(private boardService: BoardService) {}

  ngOnInit(): void {
    this.boardService.getNewGameBoard().subscribe((board: Board) => {
      this.board = board;
    });
  }

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
      this.boardService
        .applyMove(this.board?.gameId, moveIndex)
        .subscribe((board: Board) => {
          this.board = board;
        });
    }
  }
}
