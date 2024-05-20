import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BoardPegComponent } from './board-peg.component';

describe('BoardPegComponent', () => {
  let component: BoardPegComponent;
  let fixture: ComponentFixture<BoardPegComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BoardPegComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(BoardPegComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
