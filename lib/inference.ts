export type ShapeType = number[];

export type Elems<T extends any[]> = T extends [infer H, ...infer Tails]
  ? H | Elems<Tails>
  : never;

export type Head<A extends any[]> = A extends [infer H, ...infer Tails]
  ? H
  : never;
export type Tail<A extends any[]> = A extends [infer H, ...infer Tails]
  ? Tails
  : never;

export type Prev = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8];
export type Sub1<A extends number> = A extends -1 ? -1 : Prev[A];
export type Subtract<A extends number, B extends number> = B extends 0
  ? A
  : Subtract<Sub1<A>, Sub1<B>>;

//type Map<B extends any[], C> = B extends [infer H, ...infer Tails] ? MapRecurse<H, Tails, C> : [];
//type MapRecurse<H, Tails extends any[], C> = [C<H>, ...Map<Tails, C>];

export type MapSub1<A extends any[]> = A extends [infer H, ...infer Tails]
  ? H extends number
    ? [Sub1<H>, ...MapSub1<Tails>]
    : never
  : [];

export type ReduceKeepDims<
  Shape extends ShapeType,
  Axes extends number[]
> = Shape extends [infer H, ...infer Tails]
  ? Tails extends number[]
    ? 0 extends Elems<Axes>
      ? [1, ...ReduceKeepDims<Tails, MapSub1<Axes>>]
      : [H, ...ReduceKeepDims<Tails, MapSub1<Axes>>]
    : never
  : Shape extends []
  ? []
  : number[];

export type ReduceNoKeepDims<
  Shape extends ShapeType,
  Axes extends number[]
> = Shape extends [infer H, ...infer Tails]
  ? Tails extends number[]
    ? 0 extends Elems<Axes>
      ? [...ReduceNoKeepDims<Tails, MapSub1<Axes>>]
      : [H, ...ReduceNoKeepDims<Tails, MapSub1<Axes>>]
    : never
  : Shape extends []
  ? []
  : number[];

export type ReduceShape<
  Shape extends ShapeType,
  Axes extends number[] | number,
  KeepDims extends boolean
> = Axes extends number
  ? ReduceShape<Shape, [Axes], KeepDims>
  : Axes extends number[]
  ? Elems<Axes> extends never
    ? number[]
    : KeepDims extends true
    ? ReduceKeepDims<Shape, Axes>
    : KeepDims extends false
    ? ReduceNoKeepDims<Shape, Axes>
    : number[]
  : number[];

export type Reverse<A extends any[]> = A extends [infer H, ...infer Tails]
  ? [...Reverse<Tails>, H]
  : A extends []
  ? []
  : A;

export type BroadcastShape<
  A extends ShapeType,
  B extends ShapeType
> = A extends [infer HeadA, ...infer TailsA]
  ? TailsA extends number[]
    ? B extends [infer HeadB, ...infer TailsB]
      ? TailsB extends number[]
        ? HeadA extends 1
          ? [...BroadcastShape<TailsA, TailsB>, HeadB]
          : [...BroadcastShape<TailsA, TailsB>, HeadA]
        : never
      : B extends []
      ? [...TailsA, HeadA]
      : number[]
    : never
  : A extends []
  ? B
  : number[];
