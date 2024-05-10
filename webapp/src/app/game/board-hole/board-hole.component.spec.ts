import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BoardHoleComponent } from './board-hole.component';

describe('BoardHoleComponent', () => {
  let component: BoardHoleComponent;
  let fixture: ComponentFixture<BoardHoleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BoardHoleComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(BoardHoleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
