import { OrderedDict, QueryResult, SearchQuery } from "./types";

export class Dict<K, V> implements OrderedDict<K,V> {
  private dict: {[key: number]: V[]};

  constructor(private toNumber: (key: K) => number) {
    this.dict = {};
  }

  betweenBoundsFirst(query: SearchQuery<K>): QueryResult<K, V> {
    if (query.gte !== undefined) {
      const k = this.toNumber(query.gte);
      if (this.dict[k] !== undefined && this.dict[k].length > 0) {
        return [{
          key: query.gte, value: this.dict[k][this.dict[k].length - 1]
        }];
      }
      return [];
    } else if (query.lte !== undefined) {
      const k = this.toNumber(query.lte);
      if (this.dict[k] !== undefined && this.dict[k].length > 0) {
        return [{
          key: query.lte, value: this.dict[k][this.dict[k].length - 1]
        }];
      }
      return [];
    }
    return [];
  }

  deleteFirst(key: K): void {
    const k = this.toNumber(key);
    if (this.dict[k] !== undefined) {
      this.dict[k].pop();
    }
  }

  insert(key: K, value: V): void {
    const k = this.toNumber(key);
    if (this.dict[k] === undefined) {
      this.dict[k] = [];
    }
    this.dict[k].push(value);
  }
}