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

import { DataService } from './data.service';
import { Game } from '../types/game.model';
import { Move } from '../types/move.model';
import { nullMove } from '../constants/constants';

@Injectable({
  providedIn: 'root',
})
export class GameService implements OnDestroy {
  private readonly onDestroy = new Subject<void>();
  game$: Subject<Game> = new Subject<Game>();
  currentBoardIndex$: BehaviorSubject<number> = new BehaviorSubject<number>(0);
  moveToApply$: Subject<Move> = new Subject<Move>();

  constructor(private dataService: DataService) {
    this.moveToApply$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.getDraggableMoves(), this.game$),
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

  newGame(gameId: string | null, gameSize: number) {
    this.dataService.getNewGame(gameId, gameSize).subscribe((game: Game) => {
      this.game$.next(game);
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
    return combineLatest([this.game$, this.currentBoardIndex$]).pipe(
      map<[Game, number], Move>(([game, currentBoardIndex]) => {
        if (currentBoardIndex <= 0) {
          return nullMove;
        }
        return game.boards[currentBoardIndex].lastMove;
      })
    );
  }

  getDraggableMoves(): Observable<Move[]> {
    return combineLatest([this.game$, this.currentBoardIndex$]).pipe(
      map<[Game, number], Move[]>(([game, currentBoardIndex]) => {
        if (game.boards.length - 1 <= currentBoardIndex) {
          return game.boards[game.boards.length - 1].legalMoves;
        }
        return [];
      })
    );
  }
}
