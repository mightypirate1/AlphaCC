import { Component, OnDestroy, OnInit } from '@angular/core';
import {
  ReactiveFormsModule,
  FormBuilder,
  Validators,
  FormControl,
} from '@angular/forms';
import { Subject, catchError, of, switchMap, takeUntil } from 'rxjs';
import { Router } from '@angular/router';

import { DataService } from '../../services/data.service';
import { NewGameFormData } from '../../types/new-game-form-data.model';

@Component({
  selector: 'app-game-init',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './game-init.component.html',
  styleUrl: './game-init.component.scss',
})
export class GameInitComponent implements OnInit, OnDestroy {
  private readonly onDestroy = new Subject<void>();
  newGameForm = this.formBuilder.group({
    gameId: new FormControl<string | null>(null),
    gameSize: new FormControl<number>(-1, {
      validators: [Validators.required, Validators.min(0)],
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
                  this.newGameForm.controls['gameId'].setErrors({
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
          this.router.navigate([game.gameId]);
        }
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  onSubmit() {
    if (this.newGameForm.controls.gameId.value === '') {
      this.newGameForm.controls.gameId.setValue(null);
    }
    const gameId = this.newGameForm.controls.gameId.value;
    const size = +this.newGameForm.controls.gameSize.value;
    if (this.newGameForm.valid) {
      this.submit$.next({ gameId: gameId, size: size });
    }
  }
}
