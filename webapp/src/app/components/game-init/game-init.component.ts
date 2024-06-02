import {
  Component,
  EventEmitter,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import {
  ReactiveFormsModule,
  FormBuilder,
  Validators,
  FormControl,
} from '@angular/forms';
import { Subject, filter, map, takeUntil } from 'rxjs';
import { NewGameFormData } from '../../types/new-game-form-data';

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
  private submit$ = new Subject<void>();
  @Output() private formGroupChange: EventEmitter<NewGameFormData> =
    new EventEmitter<NewGameFormData>();

  constructor(private formBuilder: FormBuilder) {}

  ngOnInit(): void {
    this.submit$
      .pipe(
        takeUntil(this.onDestroy),
        filter(() => this.newGameForm.valid),
        map(() => this.newGameForm.value)
      )
      .subscribe((formData) => {
        const gameId = formData.gameId!;
        const gameSize = formData.gameSize!;
        this.formGroupChange.emit({ gameId: gameId, gameSize: gameSize });
      });
  }

  ngOnDestroy(): void {
    this.onDestroy.next();
  }

  onSubmit() {
    if (this.newGameForm.controls.gameId.value === '') {
      this.newGameForm.controls.gameId.setValue(null);
    }
    const size = this.newGameForm.controls.gameSize.value;
    this.newGameForm.controls.gameSize.setValue(+size);
    this.submit$.next();
  }
}
