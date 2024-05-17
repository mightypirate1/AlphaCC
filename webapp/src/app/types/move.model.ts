import { Point } from './point.model';

export interface Move {
  fromCoord: Point;
  toCoord: Point;
  path: Point[];
  index: number;
}
