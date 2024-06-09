import { Component, OnDestroy } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { Subject, takeUntil } from 'rxjs';

import { GameService } from '../../services/game.service';
import { BarChartComponent } from '../charts/bar-chart/bar-chart.component';

@Component({
  selector: 'app-game-plots',
  standalone: true,
  imports: [AsyncPipe, BarChartComponent],
  templateUrl: './game-plots.component.html',
  styleUrl: './game-plots.component.scss',
})
export class GamePlotsComponent implements OnDestroy {
  private readonly onDestroy = new Subject<void>();
  pi: number[] = [];
  n: number[] = [];
  q: number[] = [];

  constructor(private gameService: GameService) {
    gameService
      .getMCTSNode()
      .pipe(takeUntil(this.onDestroy))
      .subscribe((node) => {
        this.pi = node.pi;
        this.n = node.n;
        this.q = node.q;
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }
}
