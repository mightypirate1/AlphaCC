import { Routes } from '@angular/router';

import { GameComponent } from './components/game/game.component';
import { HomeComponent } from './components/home/home.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: ':gameId', component: GameComponent },
  { path: 'inspect-game', component: HomeComponent },
];
