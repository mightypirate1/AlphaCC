import { Component } from '@angular/core';

import { GameInitComponent } from '../game-init/game-init.component';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [GameInitComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {}
