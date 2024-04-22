import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Board } from './board';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class BoardService {

  boardUrl: string = 'http://localhost:8000/static-board';

  constructor(private http: HttpClient) { }

  getBoard(): Observable<Board> {
    return this.http.get<Board>(this.boardUrl);
  }
}
