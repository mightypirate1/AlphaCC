import { Injectable, OnDestroy } from '@angular/core';
import {
  BehaviorSubject,
  Observable,
  Subject,
  combineLatest,
  filter,
  map,
  of,
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
  player$: Observable<number> = of(1);

  constructor(private dataService: DataService) {
    this.moveToApply$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.getDraggableMoves(), this.game$),
        map<[Move, Move[], Game], [Move[], Game]>(
          ([moveToApply, legalMoves, game]) => [
            legalMoves.filter((legalMove) => {
              return (
                moveToApply.fromCoord.x === legalMove.fromCoord.x &&
                moveToApply.fromCoord.y === legalMove.fromCoord.y &&
                moveToApply.toCoord.x === legalMove.toCoord.x &&
                moveToApply.toCoord.y === legalMove.toCoord.y
              );
            }),
            game,
          ]
        ),
        filter(([filteredMoves]) => {
          return filteredMoves.length !== 0;
        }),
        map<[Move[], Game], [number, string]>(([filteredMoves, game]) => {
          return [filteredMoves[0].index, game.gameId];
        })
      )
      .subscribe(([moveIndex, gameId]) => {
        dataService.applyMove(gameId, moveIndex).subscribe((game) => {
          this.game$.next(game);
          this.currentBoardIndex$.next(this.currentBoardIndex$.getValue() + 1);
          if (game.boards[game.boards.length - 1].gameOver === false) {
            dataService.requestMove(gameId, 500, 100, 1).subscribe((game) => {
              this.game$.next(game);
              this.currentBoardIndex$.next(
                this.currentBoardIndex$.getValue() + 1
              );
            });
          }
        });
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  newGame(gameId: string | null, gameSize: number) {
    this.dataService
      .createNewGame(gameId, gameSize)
      .pipe(takeUntil(this.onDestroy))
      .subscribe((game: Game) => {
        this.game$.next(game);
      });
  }

  setActiveGame(gameId: string) {
    this.dataService
      .fetchGame(gameId)
      .pipe(takeUntil(this.onDestroy))
      .subscribe((game) => {
        this.currentBoardIndex$.next(game.boards.length - 1);
        this.game$.next(game);
      });
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
    return combineLatest([
      this.game$,
      this.currentBoardIndex$,
      this.player$,
    ]).pipe(
      map<[Game, number, number], Move[]>(
        ([game, currentBoardIndex, player]) => {
          if (
            game.boards.length - 1 <= currentBoardIndex &&
            game.boards[currentBoardIndex].currentPlayer === player &&
            game.boards[currentBoardIndex].gameOver === false
          ) {
            return game.boards[game.boards.length - 1].legalMoves;
          }
          return [];
        }
      )
    );
  }
}
