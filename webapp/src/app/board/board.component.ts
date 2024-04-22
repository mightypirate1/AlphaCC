import { Component, OnInit } from '@angular/core';
import { BoardService } from '../board.service';
import { Board } from '../board';

@Component({
  selector: 'app-board',
  standalone: true,
  imports: [],
  templateUrl: './board.component.html',
  styleUrl: './board.component.css'
})
export class BoardComponent implements OnInit{

  board: Board | undefined;
  message: string = '';
  matrix: number[][] | undefined;

  constructor(private boardService: BoardService) { }

  ngOnInit(): void {
    this.showBoard();
  }

  showBoard() {
    this.boardService.getBoard()
      .subscribe({
        next: (data) => {
          this.board = data;
          this.message = data.message;
          this.matrix = data.matrix;
          console.log(this.board);
        }
      })
    console.log(this.board);
  }
}
