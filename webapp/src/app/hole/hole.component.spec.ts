import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HoleComponent } from './hole.component';

describe('HoleComponent', () => {
  let component: HoleComponent;
  let fixture: ComponentFixture<HoleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HoleComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(HoleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
