import { Component, OnDestroy } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { Subject, takeUntil } from 'rxjs';
import { CdkDragDrop, DragDropModule } from '@angular/cdk/drag-drop';

import { GameService } from '../../services/game.service';
import { BoardPegComponent } from '../board-peg/board-peg.component';
import { Move } from '../../types/move.model';
import { Point } from '../../types/point.model';
import { nullMove, nullPoint } from '../../constants/constants';

@Component({
  selector: 'app-game-board',
  standalone: true,
  templateUrl: './game-board.component.html',
  styleUrl: './game-board.component.scss',
  imports: [AsyncPipe, CommonModule, DragDropModule, BoardPegComponent],
})
export class GameBoardComponent implements OnDestroy {
  colors = ['', 'orange', 'rebeccapurple'];
  lastMove: Move = nullMove;
  selected: Point = nullPoint;
  draggableMoves: Move[] = [];
  board: number[][] = [];

  private readonly onDestroy = new Subject<void>();

  constructor(private gameService: GameService) {
    gameService
      .getCurrentBoardMatrix()
      .subscribe((board) => (this.board = board));

    gameService
      .getLastMove()
      .pipe(takeUntil(this.onDestroy))
      .subscribe((move) => {
        if (move !== null) {
          this.lastMove = move;
        }
      });
    gameService
      .getDraggableMoves()
      .pipe(takeUntil(this.onDestroy))
      .subscribe((moves) => {
        this.draggableMoves = moves;
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  setSelected(x: number, y: number): void {
    this.selected = { x: x, y: y };
  }

  isLegalTarget(targetX: number, targetY: number): boolean {
    return this.draggableMoves.some(
      (move) =>
        move.toCoord.x === targetX &&
        move.toCoord.y === targetY &&
        move.fromCoord.x === this.selected.x &&
        move.fromCoord.y === this.selected.y
    );
  }

  isLegalSource(sourceX: number, sourceY: number): boolean {
    return this.draggableMoves.some(
      (move) => move.fromCoord.x === sourceX && move.fromCoord.y === sourceY
    );
  }

  drop($event: CdkDragDrop<{ row: number; col: number }>): void {
    const fromX = $event.item.dropContainer.data.row;
    const fromY = $event.item.dropContainer.data.col;
    const toX = $event.container.data.row;
    const toY = $event.container.data.col;

    const move: Move = {
      fromCoord: { x: fromX, y: fromY },
      toCoord: { x: toX, y: toY },
      path: [],
      index: -1,
    };

    if (this.isLegalSource(fromX, fromY) && this.isLegalTarget(toX, toY)) {
      const movedValue = this.board[fromX][fromY];
      this.board[fromX][fromY] = this.board[toX][toY];
      this.board[toX][toY] = movedValue;
      this.gameService.applyMove(move);
    }
  }
}
