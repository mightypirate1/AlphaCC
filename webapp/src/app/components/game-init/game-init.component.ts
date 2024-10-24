import { Component, OnDestroy, OnInit } from '@angular/core';
import { JsonPipe } from '@angular/common';
import { Router } from '@angular/router';
import { ReactiveFormsModule, FormBuilder, Validators } from '@angular/forms';
import { Subject, catchError, of, switchMap, takeUntil } from 'rxjs';

import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatRadioModule } from '@angular/material/radio';
import { MatSelectModule } from '@angular/material/select';

import { DataService } from '../../services/data.service';

@Component({
  selector: 'app-game-init',
  standalone: true,
  imports: [
    JsonPipe,
    MatButtonModule,
    MatCardModule,
    MatDividerModule,
    MatFormFieldModule,
    MatInputModule,
    MatRadioModule,
    MatSelectModule,
    ReactiveFormsModule,
  ],
  templateUrl: './game-init.component.html',
  styleUrl: './game-init.component.scss',
})
export class GameInitComponent implements OnInit, OnDestroy {
  private readonly onDestroy = new Subject<void>();

  form = this.formBuilder.group({
    gameId: this.formBuilder.control(''),
    boardSize: this.formBuilder.control(-1, {
      validators: [Validators.required, Validators.min(0)],
      nonNullable: true,
    }),
    firstPlayer: this.formBuilder.control('HUMAN', {
      validators: [Validators.required],
      nonNullable: true,
    }),
    secondPlayer: this.formBuilder.control('AI', {
      validators: [Validators.required],
      nonNullable: true,
    }),
  });
  private submit$ = new Subject<void>();

  constructor(
    private formBuilder: FormBuilder,
    private dataService: DataService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.submit$
      .pipe(
        takeUntil(this.onDestroy),
        switchMap(() => {
          const gameId = this.form.controls.gameId.value;
          const size = +this.form.controls.boardSize.value;
          return this.dataService.createNewGame(gameId, size).pipe(
            catchError((err) => {
              if (err.status === 400) {
                this.form.controls['gameId'].setErrors({
                  incorrect: true,
                });
              }
              return of(null);
            })
          );
        })
      )
      .subscribe((game) => {
        if (game) {
          const firstPlayer = this.form.controls.firstPlayer.value;
          const secondPlayer = this.form.controls.secondPlayer.value;
          this.router.navigate([
            game.gameId,
            {
              players: [firstPlayer, secondPlayer],
            },
          ]);
        }
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  onReset() {
    this.form.reset;
  }

  onSubmit() {
    if (this.form.controls.gameId.value === '') {
      this.form.controls.gameId.setValue(null);
    }
    if (this.form.valid) {
      this.submit$.next();
    }
  }
}
