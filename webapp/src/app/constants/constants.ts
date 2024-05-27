import { Point } from '@angular/cdk/drag-drop';
import { Move } from '../types/move.model';

export const nullPoint: Point = { x: -1, y: -1 };

export const nullMove: Move = {
  fromCoord: nullPoint,
  toCoord: nullPoint,
  path: [],
  index: -1,
};
