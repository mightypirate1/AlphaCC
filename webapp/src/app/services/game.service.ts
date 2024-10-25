import { Injectable, OnDestroy } from '@angular/core';
import {
  BehaviorSubject,
  NEVER,
  Observable,
  Subject,
  catchError,
  combineLatest,
  filter,
  map,
  of,
  switchMap,
  takeUntil,
  timer,
  withLatestFrom,
} from 'rxjs';

import { DataService } from './data.service';
import { Game } from '../types/game.model';
import { Move } from '../types/move.model';
import { nullMove } from '../constants/constants';
import { MCTSNode } from '../types/mcts-node.model';

@Injectable({ providedIn: 'root' })
export class GameService implements OnDestroy {
  private readonly onDestroy = new Subject<void>();

  private game$: Subject<Game> = new Subject<Game>();
  private playerSettings$: BehaviorSubject<string[]> = new BehaviorSubject<
    string[]
  >([]);
  private currentBoardIndex$: BehaviorSubject<number> =
    new BehaviorSubject<number>(0);
  private moveToApply$: Subject<Move> = new Subject<Move>();
  private showMode$: BehaviorSubject<boolean> = new BehaviorSubject<boolean>(
    false
  );

  constructor(private dataService: DataService) {
    this.moveToApply$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.draggableMoves(), this.game$),
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
        });
      });
    this.showMode$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.game$),
        switchMap(([showMode, game]) => {
          return (
            showMode
              ? dataService.showModeOn(game.gameId)
              : dataService.showModeOff(game.gameId)
          ).pipe(catchError(() => of({})));
        })
      )
      .subscribe();

    this.game$
      .pipe(
        takeUntil(this.onDestroy),
        withLatestFrom(this.playerSettings$),
        switchMap(([game, settings]) => {
          const player = game.boards[game.boards.length - 1].currentPlayer;
          if (
            settings[player - 1] === 'AI' &&
            game.boards[game.boards.length - 1].gameOver === false
          ) {
            return dataService.requestMove(game.gameId, 500, 100, 1);
          } else return of(null);
        })
      )
      .subscribe((game) => {
        if (game !== null) {
          this.game$.next(game);
          this.currentBoardIndex$.next(this.currentBoardIndex$.getValue() + 1);
        }
      });
  }

  ngOnDestroy(): void {
    console.log('GameService destroyed');
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

  setPlayersSettings(players: string[]) {
    if (
      players.every((val) => val === 'HUMAN' || val === 'AI') &&
      players.length >= 2
    ) {
      this.playerSettings$.next(players);
    } else {
      this.playerSettings$.next(['HUMAN', 'HUMAN']);
    }
  }

  applyMove(move: Move): void {
    this.moveToApply$.next(move);
  }

  changeCurrentMoveIndex(newCurrentBoardIndex: number) {
    this.currentBoardIndex$.next(newCurrentBoardIndex);
  }

  currentBoardMatrix(): Observable<number[][]> {
    return combineLatest([this.game$, this.currentBoardIndex$]).pipe(
      map<[Game, number], number[][]>(([game, boardIndex]): number[][] => {
        return game.boards[boardIndex].matrix;
      })
    );
  }

  lastMove(): Observable<Move> {
    return combineLatest([this.game$, this.currentBoardIndex$]).pipe(
      map<[Game, number], Move>(([game, currentBoardIndex]) => {
        if (currentBoardIndex <= 0) {
          return nullMove;
        }
        return game.boards[currentBoardIndex].lastMove;
      })
    );
  }

  draggableMoves(): Observable<Move[]> {
    return combineLatest([
      this.game$,
      this.currentBoardIndex$,
      this.playerSettings$,
    ]).pipe(
      map<[Game, number, string[]], Move[]>(
        ([game, currentBoardIndex, playerSettings]) => {
          if (
            game.boards.length - 1 <= currentBoardIndex &&
            playerSettings[game.boards[currentBoardIndex].currentPlayer - 1] ===
              'HUMAN' &&
            game.boards[currentBoardIndex].gameOver === false
          ) {
            return game.boards[game.boards.length - 1].legalMoves;
          }
          return [];
        }
      )
    );
  }

  pollMCTSNode(): Observable<MCTSNode> {
    return this.showMode$.pipe(
      switchMap((showMode) => (showMode ? timer(1, 3000) : NEVER)),
      switchMap(() => this.currentBoardIndex$),
      withLatestFrom(this.game$),
      switchMap<[number, Game], Observable<MCTSNode>>(([boardIndex, game]) => {
        return this.dataService.fetchMCTSNode(game.gameId, boardIndex);
      })
    );
  }

  showMode(): Observable<boolean> {
    return this.showMode$.asObservable();
  }

  currentBoardIndex(): Observable<number> {
    return this.currentBoardIndex$.asObservable();
  }

  game(): Observable<Game> {
    return this.game$.asObservable();
  }

  toggleShowMode(): void {
    this.showMode$.next(!this.showMode$.getValue());
  }

  setShowModeOff(): void {
    this.showMode$.next(false);
  }
}
