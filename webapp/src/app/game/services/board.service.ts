import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Board } from '../types/board.model';
import { Observable, map } from 'rxjs';
import { environment } from '../../../environments/environment.development';
import { BoardIO } from '../types/board-io';

@Injectable({
  providedIn: 'root',
})
export class BoardService {
  staticBoardUrl: string = environment.backendUrl + '/static-board';
  newGameUrl: string = environment.backendUrl + '/new-game';
  applyMoveUrl: string = environment.backendUrl + '/apply-move';

  constructor(private http: HttpClient) {}

  getStaticBoard(): Observable<Board> {
    return this.http
      .get<BoardIO>(this.staticBoardUrl)
      .pipe(map((boardIo) => new Board(boardIo)));
  }

  getNewGameBoard(): Observable<Board> {
    return this.http
      .post<BoardIO>(this.newGameUrl, { size: 9, gameId: null })
      .pipe(map((boardIo) => new Board(boardIo)));
  }

  applyMove(gameId: string, moveIndex: number): Observable<Board> {
    return this.http
      .post<BoardIO>(this.applyMoveUrl, {
        gameId: gameId,
        moveIndex: moveIndex,
      })
      .pipe(map((boardIo) => new Board(boardIo)));
  }
}
