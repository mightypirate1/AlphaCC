import { Component, OnDestroy, OnInit } from '@angular/core';
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
import { NewGameFormData } from '../../types/new-game-form-data.model';

@Component({
  selector: 'app-game-init',
  standalone: true,
  imports: [
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
    gameSize: this.formBuilder.control(-1, {
      validators: [Validators.required, Validators.min(0)],
      nonNullable: true,
    }),
    player: this.formBuilder.control(-1, {
      validators: [Validators.required, Validators.min(0), Validators.max(2)],
      nonNullable: true,
    }),
  });
  private submit$ = new Subject<NewGameFormData>();

  constructor(
    private formBuilder: FormBuilder,
    private dataService: DataService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.submit$
      .pipe(
        takeUntil(this.onDestroy),
        switchMap((formData) => {
          return this.dataService
            .createNewGame(formData.gameId, formData.size)
            .pipe(
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
          this.router.navigate([
            game.gameId,
            { player: this.form.controls.player.value },
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
    const gameId = this.form.controls.gameId.value;
    const size = +this.form.controls.gameSize.value;
    const player = +this.form.controls.player.value;
    if (this.form.valid) {
      this.submit$.next({ gameId: gameId, size: size, player: player });
    }
  }
}
