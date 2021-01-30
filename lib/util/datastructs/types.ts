export interface SearchQuery<K> {
  gt?: K;
  gte?: K;
  lt?: K;
  lte?: K;
}

export type QueryResult<K,V> = {key: K, value: V}[];

export interface OrderedDict<K, V> {
  betweenBoundsFirst(query: SearchQuery<K>): QueryResult<K, V>;

  deleteFirst(key: K): void;

  insert(key: K, value: V): void;
}