import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Board } from './board';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment.development';

@Injectable({
  providedIn: 'root'
})
export class BoardService {

  boardUrl: string = environment.backendUrl + '/static-board';

  constructor(private http: HttpClient) { }

  getBoard(): Observable<Board> {
    return this.http.get<Board>(this.boardUrl);
  }
}
