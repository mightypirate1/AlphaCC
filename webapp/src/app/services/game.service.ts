import { Injectable, OnDestroy } from '@angular/core';
import {
  BehaviorSubject,
  Observable,
  Subject,
  combineLatest,
  filter,
  map,
  takeUntil,
  withLatestFrom,
} from 'rxjs';

import { Game } from '../types/game.model';
import { DataService } from './data.service';
import { Move } from '../types/move.model';

@Injectable({
  providedIn: 'root',
})
export class GameService implements OnDestroy {
  private readonly onDestroy = new Subject<void>();
  game$: Subject<Game>;
  currentBoardIndex$: BehaviorSubject<number>;
  moveToApply$: Subject<Move> = new Subject<Move>();

  constructor(private dataService: DataService) {
    this.game$ = new Subject<Game>();
    this.currentBoardIndex$ = new BehaviorSubject<number>(0);
    this.currentBoardIndex$.next(0);

    this.dataService.getNewGame().subscribe((game: Game) => {
      this.game$.next(game);
    });

    this.moveToApply$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.getLegalMoves(), this.game$),
        map<[Move, Move[], Game], [Move[], Game]>(
          ([moveToApply, legalMoves, game]) => [
            legalMoves.filter((move) => {
              return (
                move.fromCoord.x === moveToApply.fromCoord.x &&
                move.fromCoord.y === moveToApply.fromCoord.y &&
                move.toCoord.x === moveToApply.toCoord.x &&
                move.toCoord.y === moveToApply.toCoord.y
              );
            }),
            game,
          ]
        ),
        filter(([filteredMoves]) => {
          return filteredMoves.length !== 0;
        }),
        map<[Move[], Game], [number, string]>(([move, game]) => {
          return [move[0].index, game.gameId];
        })
      )
      .subscribe(([moveIndex, gameId]) => {
        dataService.applyMove(gameId, moveIndex).subscribe((game) => {
          this.game$.next(game);
          this.currentBoardIndex$.next(this.currentBoardIndex$.getValue() + 1);
        });
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  applyMove(move: Move): void {
    this.moveToApply$.next(move);
  }

  changeCurrentMoveIndex(newCurrentBoardIndex: number) {
    this.currentBoardIndex$.next(newCurrentBoardIndex);
  }

  getCurrentBoardMatrix(): Observable<number[][]> {
    return combineLatest([this.game$, this.currentBoardIndex$]).pipe(
      map<[Game, number], number[][]>(([game, boardIndex]): number[][] => {
        return game.boards[boardIndex].matrix;
      })
    );
  }

  getLastMove(): Observable<Move> {
    return this.game$.pipe(
      map((game) => game.boards[game.boards.length - 1].lastMove)
    );
  }

  getLegalMoves(): Observable<Move[]> {
    return this.game$.pipe(
      map((game) => game.boards[game.boards.length - 1].legalMoves)
    );
  }
}
