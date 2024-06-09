import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GamePlotsComponent } from './game-plots.component';

describe('GamePlotsComponent', () => {
  let component: GamePlotsComponent;
  let fixture: ComponentFixture<GamePlotsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GamePlotsComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(GamePlotsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
