import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GameInitComponent } from './game-init.component';

describe('GameInitComponent', () => {
  let component: GameInitComponent;
  let fixture: ComponentFixture<GameInitComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GameInitComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(GameInitComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
