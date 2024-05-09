import { Component, Input, OnInit } from '@angular/core';
import { Point } from '../point';

@Component({
    selector: 'app-hole',
    standalone: true,
    templateUrl: './hole.component.html',
    styleUrl: './hole.component.css',
    imports: []
})
export class HoleComponent implements OnInit {
  
  @Input() pegType: number | undefined;
  @Input() point: Point | undefined;

  // @Output() moveFrom = new EventEmitter<Point>();
  // @Output() moveTo: Point;

  color: string = '';

  ngOnInit(): void {
    switch(this.pegType) {
      case 1: {
        this.color = "#6b5b95";
        break;
      }
      case 2: {
        this.color = "#b2ad7f";
        break;
      }
      default: {
        this.color = "#f0f0f0";
      }
    }
  }

  onClick() {
    this.color = "#000000";
    console.log("click! x=" + this.point?.x + " y=" + this.point?.y);
  }

  onMouseUp() {
    console.log("MouseUp: x=" + this.point?.x + ", y=" + this.point?.y);
  }
  onMouseDown() {
    console.log("MouseDown: x=" + this.point?.x + ", y=" + this.point?.y);
  }

  onKeyup() {
    console.log("keyup!");
  }
}
