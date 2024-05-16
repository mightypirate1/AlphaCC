import { Directive, ElementRef, Input, OnInit } from '@angular/core';

@Directive({
  selector: '[appPeg]',
  standalone: true,
})
export class PegDirective implements OnInit {
  @Input() appPeg: string = '';

  constructor(private el: ElementRef) {}

  ngOnInit(): void {
    this.el.nativeElement.style.backgroundColor = this.appPeg;
  }
}
