import { Component, OnInit } from '@angular/core';
import { BoardService } from '../shared/board.service';
import { Board } from '../shared/board.model';
import { HoleComponent } from '../board-hole/board-hole.component';

@Component({
    selector: 'app-board',
    standalone: true,
    templateUrl: './board.component.html',
    styleUrl: './board.component.scss',
    imports: [HoleComponent]
})
export class BoardComponent implements OnInit{

  board: Board | undefined;

  constructor(private boardService: BoardService) { }

  ngOnInit(): void {
    this.showBoard();
  }

  showBoard() {
    this.boardService.getBoard()
      .subscribe({
        next: (data) => {
          this.board = data;
        }
      })
  }
}
