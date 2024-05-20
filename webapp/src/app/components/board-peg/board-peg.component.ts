import { Component, HostBinding, Input } from '@angular/core';

@Component({
  selector: 'app-board-peg',
  standalone: true,
  imports: [],
  templateUrl: './board-peg.component.html',
  styleUrl: './board-peg.component.scss',
})
export class BoardPegComponent {
  @HostBinding('style.background-color') @Input() pegColor: string = '';
}
