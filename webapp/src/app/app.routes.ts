import { Routes } from '@angular/router';
import { GameComponent } from './components/game/game.component';

export const routes: Routes = [
  { path: '', redirectTo: '/home', pathMatch: 'full' },
  { path: 'home', component: GameComponent },
  { path: 'new-game', component: GameComponent },
  { path: 'inspect-game', component: GameComponent },
  { path: 'about', component: GameComponent },
];
