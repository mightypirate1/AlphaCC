import { Component, OnInit } from '@angular/core';
import { BoardService } from '../board.service';
import { Board } from '../board';
import { HoleComponent } from "../hole/hole.component";

@Component({
    selector: 'app-board',
    standalone: true,
    templateUrl: './board.component.html',
    styleUrl: './board.component.css',
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
