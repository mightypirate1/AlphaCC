import { Routes } from '@angular/router';
import { BoardComponent } from './game/board/board.component';

export const routes: Routes = [
  { path: '', redirectTo: '/home', pathMatch: 'full' },
  { path: 'home', component: BoardComponent },
  { path: 'new-game', component: BoardComponent },
  { path: 'inspect-game', component: BoardComponent },
  { path: 'about', component: BoardComponent },
];
