import { MCTSNodeIO } from './mcts-node-io.model';

export class MCTSNode {
  pi: number[];
  vHat: number;
  n: number[];
  q: number[];

  constructor(nodeIo: MCTSNodeIO) {
    this.pi = nodeIo.pi;
    this.vHat = nodeIo.vHat;
    this.n = nodeIo.n;
    this.q = nodeIo.q;
  }
}
