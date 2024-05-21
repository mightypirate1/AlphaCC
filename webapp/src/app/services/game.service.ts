import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, map } from 'rxjs';
import { environment } from '../../environments/environment.development';
import { GameIO } from '../types/game-io';
import { Game } from '../types/game.model';

@Injectable({
  providedIn: 'root',
})
export class GameService {
  newGameUrl: string = environment.backendUrl + '/new-game';
  applyMoveUrl: string = environment.backendUrl + '/apply-move';

  constructor(private http: HttpClient) {}

  getNewGameBoard(): Observable<Game> {
    return this.http
      .post<GameIO>(this.newGameUrl, { size: 9, gameId: null })
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
}
