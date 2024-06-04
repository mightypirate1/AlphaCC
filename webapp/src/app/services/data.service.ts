import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';

import { environment } from '../../environments/environment.development';
import { GameIO } from '../types/game-io.model';
import { Game } from '../types/game.model';

@Injectable({
  providedIn: 'root',
})
export class DataService {
  private newGameUrl: string = environment.backendUrl + '/new-game';
  private fetchGameUrl: string = environment.backendUrl + '/fetch-game';
  private applyMoveUrl: string = environment.backendUrl + '/apply-move';
  private requestMoveUrl: string = environment.backendUrl + '/request-move';

  constructor(private http: HttpClient) {}

  createNewGame(gameId: string | null, size: number): Observable<Game> {
    return this.http
      .post<GameIO>(this.newGameUrl, { gameId: gameId, size: size })
      .pipe(map((gameIo) => new Game(gameIo)));
  }

  fetchGame(gameId: string): Observable<Game> {
    return this.http
      .get<GameIO>(this.fetchGameUrl + '?game_id=' + gameId)
      .pipe(map((gameIo) => new Game(gameIo)));
  }

  applyMove(gameId: string, moveIndex: number): Observable<Game> {
    return this.http
      .post<GameIO>(this.applyMoveUrl, {
        gameId: gameId,
        moveIndex: moveIndex,
      })
      .pipe(map((gameIo) => new Game(gameIo)));
  }

  requestMove(
    gameId: string,
    nRollouts: number,
    rolloutDepth: number,
    temperature: number
  ): Observable<Game> {
    return this.http
      .post<GameIO>(this.requestMoveUrl, {
        gameId: gameId,
        nRollouts: nRollouts,
        rolloutDepth: rolloutDepth,
        temperature: temperature,
      })
      .pipe(map((gameIo) => new Game(gameIo)));
  }
}
