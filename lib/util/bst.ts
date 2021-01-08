export type Comparison<K> = (a: K, b: K) => number;

export type EqComparison<V> = (a: V, b: V) => boolean;

export type QueryResult<K,V> = {key: K, value: V}[];

export interface BSTOptions<K, V> {
  unique?: boolean;
  key?: K;
  value?: V;
  compareKeys?: Comparison<K>;
  compareValues?: EqComparison<V>;
  parent?: BinarySearchTree<K,V>;
}

export interface SearchQuery<K> {
  gt?: K;
  gte?: K;
  lt?: K;
  lte?: K;
}

/*
 * Default compareKeys function will work for numbers, strings and dates
 */
export function defaultCompareKeysFunction<K> (a: K, b: K) {
  if (a < b) { return -1; }
  if (a > b) { return 1; }
  if (a === b) { return 0; }

  var err = new Error("Couldn't compare elements");
  throw err;
}

// Append all elements in toAppend to array
function append<V>(array: V[], toAppend: V[]) {
  for (let i = 0; i < toAppend.length; i++) {
    array.push(toAppend[i]);
  }
}

export class BinarySearchTree<K,V> {
  protected left?: BinarySearchTree<K, V>;
  protected right?: BinarySearchTree<K, V>;
  protected parent?: BinarySearchTree<K, V>;

  public key: K;
  public value: V[];

  protected unique: boolean;

  protected compareKeys: Comparison<K>;
  protected compareValues: EqComparison<V>;

  constructor(options?: BSTOptions<K,V>) {
    options = options || {};

    this.left = null;
    this.right = null;
    this.parent = options.parent !== undefined ? options.parent : null;
    if (options.hasOwnProperty('key')) { this.key = options.key; }
    this.value = options.hasOwnProperty('value') ? [options.value] : [];
    this.unique = options.unique || false;
  
    this.compareKeys = options.compareKeys || defaultCompareKeysFunction;
    this.compareValues = options.compareValues || ((a: V, b: V) => a === b);
  }

  getMaxKeyDescendant(): BinarySearchTree<K,V> {
    if (this.right) {
      return this.right.getMaxKeyDescendant();
    } else {
      return this;
    }
  }

  getMaxKey() {
    return this.getMaxKeyDescendant().key;
  }

  getMinKeyDescendant(): BinarySearchTree<K, V> {
    if (this.left) {
      return this.left.getMinKeyDescendant()
    } else {
      return this;
    }
  }

  getMinKey() {
    return this.getMinKeyDescendant().key;
  }

  checkAllNodesFullfillCondition(test: (k: K, v: V[]) => boolean) {
    if (!this.hasOwnProperty('key')) { return; }
  
    test(this.key, this.value);
    if (this.left) { this.left.checkAllNodesFullfillCondition(test); }
    if (this.right) { this.right.checkAllNodesFullfillCondition(test); }
  }

  checkNodeOrdering() {  
    if (!this.hasOwnProperty('key')) { return; }
  
    if (this.left) {
      this.left.checkAllNodesFullfillCondition((k: K, _v: V[]) => {
        if (this.compareKeys(k, this.key) >= 0) {
          throw new Error('Tree with root ' + this.key + ' is not a binary search tree');
        }
        return true;
      });
      this.left.checkNodeOrdering();
    }
  
    if (this.right) {
      this.right.checkAllNodesFullfillCondition((k: K, _v: V[]) => {
        if (this.compareKeys(k, this.key) <= 0) {
          throw new Error('Tree with root ' + this.key + ' is not a binary search tree');
        }
        return true;
      });
      this.right.checkNodeOrdering();
    }
  }

  checkInternalPointers() {
    if (this.left) {
      if (this.left.parent !== this) { throw new Error('Parent pointer broken for key ' + this.key); }
      this.left.checkInternalPointers();
    }
  
    if (this.right) {
      if (this.right.parent !== this) { throw new Error('Parent pointer broken for key ' + this.key); }
      this.right.checkInternalPointers();
    }
  }

  checkIsBST() {
    this.checkNodeOrdering();
    this.checkInternalPointers();
    if (this.parent) { throw new Error("The root shouldn't have a parent"); }
  }

  getNumberOfKeys() {
    var res;
  
    if (!this.hasOwnProperty('key')) { return 0; }
  
    res = 1;
    if (this.left) { res += this.left.getNumberOfKeys(); }
    if (this.right) { res += this.right.getNumberOfKeys(); }
  
    return res;
  }

  createSimilar(options: BSTOptions<K, V>): BinarySearchTree<K, V> {
    options = options || {};
    options.unique = this.unique;
    options.compareKeys = this.compareKeys;
  
    return new BinarySearchTree(options);
  }

  createLeftChild(options: BSTOptions<K, V>) {
    var leftChild = this.createSimilar(options);
    leftChild.parent = this;
    this.left = leftChild;
  
    return leftChild;
  }

  createRightChild(options: BSTOptions<K, V>) {
    var rightChild = this.createSimilar(options);
    rightChild.parent = this;
    this.right = rightChild;
  
    return rightChild;
  }

  insert(key: K, value: V) {
    // Empty tree, insert as root
    if (!this.hasOwnProperty('key')) {
      this.key = key;
      this.value.push(value);
      return;
    }
  
    // Same key as root
    if (this.compareKeys(this.key, key) === 0) {
      if (this.unique) {
        var err = new Error("Can't insert key " + key + ", it violates the unique constraint");
        throw err;
      } else {
        this.value.push(value);
      }
      return;
    }
  
    if (this.compareKeys(key, this.key) < 0) {
      // Insert in left subtree
      if (this.left) {
        this.left.insert(key, value);
      } else {
        this.createLeftChild({ key: key, value: value });
      }
    } else {
      // Insert in right subtree
      if (this.right) {
        this.right.insert(key, value);
      } else {
        this.createRightChild({ key: key, value: value });
      }
    }
  }

  search(key: K): V[] {
    if (!this.hasOwnProperty('key')) { return []; }
  
    if (this.compareKeys(this.key, key) === 0) { return this.value; }
  
    if (this.compareKeys(key, this.key) < 0) {
      if (this.left) {
        return this.left.search(key);
      } else {
        return [];
      }
    } else {
      if (this.right) {
        return this.right.search(key);
      } else {
        return [];
      }
    }
  }

  getLowerBoundMatcher(query: SearchQuery<K>) {
    // No lower bound
    if (!query.hasOwnProperty('gt') && !query.hasOwnProperty('gte')) {
      return () => { return true; };
    }
  
    if (query.hasOwnProperty('gt') && query.hasOwnProperty('gte')) {
      if (this.compareKeys(query.gte, query.gt) === 0) {
        return (key: K) => { return this.compareKeys(key, query.gt) > 0; };
      }
  
      if (this.compareKeys(query.gte, query.gt) > 0) {
        return (key: K) => { return this.compareKeys(key, query.gte) >= 0; };
      } else {
        return (key: K) => { return this.compareKeys(key, query.gt) > 0; };
      }
    }
  
    if (query.hasOwnProperty('gt')) {
      return (key: K) => { return this.compareKeys(key, query.gt) > 0; };
    } else {
      return (key: K) => { return this.compareKeys(key, query.gte) >= 0; };
    }
  }

  getUpperBoundMatcher(query: SearchQuery<K>) {
    // No lower bound
    if (!query.hasOwnProperty('lt') && !query.hasOwnProperty('lte')) {
      return () => { return true; };
    }
  
    if (query.hasOwnProperty('lt') && query.hasOwnProperty('lte')) {
      if (this.compareKeys(query.lte, query.lt) === 0) {
        return (key: K) => { return this.compareKeys(key, query.lt) < 0; };
      }
  
      if (this.compareKeys(query.lte, query.lt) < 0) {
        return (key: K) => { return this.compareKeys(key, query.lte) <= 0; };
      } else {
        return (key: K) => { return this.compareKeys(key, query.lt) < 0; };
      }
    }
  
    if (query.hasOwnProperty('lt')) {
      return (key: K) => { return this.compareKeys(key, query.lt) < 0; };
    } else {
      return (key: K) => { return this.compareKeys(key, query.lte) <= 0; };
    }
  }

  betweenBounds(query: SearchQuery<K>, lbm?: (key: K) => boolean, ubm?: (key: K) => boolean) {
    var res: QueryResult<K,V> = [];
  
    if (!this.hasOwnProperty('key')) { return []; }   // Empty tree
  
    lbm = lbm || this.getLowerBoundMatcher(query);
    ubm = ubm || this.getUpperBoundMatcher(query);
  
    if (lbm(this.key) && this.left) { append(res, this.left.betweenBounds(query, lbm, ubm)); }
    if (lbm(this.key) && ubm(this.key)) { 
      for (let i = 0; i < this.value.length; i++) {
        res.push({key: this.key, value: this.value[i]});
      }
    }
    if (ubm(this.key) && this.right) { append(res, this.right.betweenBounds(query, lbm, ubm)); }
  
    return res;
  }

  betweenBoundsFirst(query: SearchQuery<K>, lbm?: (key: K) => boolean, ubm?: (key: K) => boolean): QueryResult<K, V> {
    if (!this.hasOwnProperty('key')) { return []; }   // Empty tree
  
    lbm = lbm || this.getLowerBoundMatcher(query);
    ubm = ubm || this.getUpperBoundMatcher(query);
  
    if (lbm(this.key) && this.left) {
      const res = this.left.betweenBoundsFirst(query, lbm, ubm);
      if (res.length > 0) {
        return res;
      };
    }
    if (lbm(this.key) && ubm(this.key)) { 
      if (this.value.length > 0) {
        return [{key: this.key, value: this.value[0]}];
      }
    }
    if (ubm(this.key) && this.right) {
      return this.right.betweenBoundsFirst(query, lbm, ubm);
    }

    return [];
  }

  deleteIfLeaf() {
    if (this.left || this.right) { return false; }
  
    // The leaf is itself a root
    if (!this.parent) {
      delete this.key;
      this.value = [];
      return true;
    }
  
    if (this.parent.left === this) {
      this.parent.left = null;
    } else {
      this.parent.right = null;
    }
  
    return true;
  }

  deleteIfOnlyOneChild() {
    var child: BinarySearchTree<K, V>;
  
    if (this.left && !this.right) { child = this.left; }
    if (!this.left && this.right) { child = this.right; }
    if (!child) { return false; }
  
    // Root
    if (!this.parent) {
      this.key = child.key;
      this.value = child.value;
  
      this.left = null;
      if (child.left) {
        this.left = child.left;
        child.left.parent = this;
      }
  
      this.right = null;
      if (child.right) {
        this.right = child.right;
        child.right.parent = this;
      }
  
      return true;
    }
  
    if (this.parent.left === this) {
      this.parent.left = child;
      child.parent = this.parent;
    } else {
      this.parent.right = child;
      child.parent = this.parent;
    }
  
    return true;
  }

  delete(key: K, value: V) {
    let newValues: V[] = [];
    let replaceWith;
  
    if (!this.hasOwnProperty('key')) { return; }
  
    if (this.compareKeys(key, this.key) < 0) {
      if (this.left) { this.left.delete(key, value); }
      return;
    }
  
    if (this.compareKeys(key, this.key) > 0) {
      if (this.right) { this.right.delete(key, value); }
      return;
    }
  
    if (!(this.compareKeys(key, this.key) === 0)) { return; }
  
    // Delete only a value
    if (this.value.length > 1 && value !== undefined) {
      this.value.forEach((d) => {
        if (!this.compareValues(d, value)) { newValues.push(d); }
      });
      this.value = newValues;
      return;
    }
  
    // Delete the whole node
    if (this.deleteIfLeaf()) {
      return;
    }
    if (this.deleteIfOnlyOneChild()) {
      return;
    }
  
    // We are in the case where the node to delete has two children
    if (Math.random() >= 0.5) {   // Randomize replacement to avoid unbalancing the tree too much
      // Use the in-order predecessor
      replaceWith = this.left.getMaxKeyDescendant();
  
      this.key = replaceWith.key;
      this.value = replaceWith.value;
  
      if (this === replaceWith.parent) {   // Special case
        this.left = replaceWith.left;
        if (replaceWith.left) { replaceWith.left.parent = replaceWith.parent; }
      } else {
        replaceWith.parent.right = replaceWith.left;
        if (replaceWith.left) { replaceWith.left.parent = replaceWith.parent; }
      }
    } else {
      // Use the in-order successor
      replaceWith = this.right.getMinKeyDescendant();
  
      this.key = replaceWith.key;
      this.value = replaceWith.value;
  
      if (this === replaceWith.parent) {   // Special case
        this.right = replaceWith.right;
        if (replaceWith.right) { replaceWith.right.parent = replaceWith.parent; }
      } else {
        replaceWith.parent.left = replaceWith.right;
        if (replaceWith.right) { replaceWith.right.parent = replaceWith.parent; }
      }
    }
  }

  executeOnEveryNode(fn: (tree: BinarySearchTree<K, V>) => void) {
    if (this.left) { this.left.executeOnEveryNode(fn); }
    fn(this);
    if (this.right) { this.right.executeOnEveryNode(fn); }
  }

  prettyPrint(printData: boolean, spacing?: string) {
    spacing = spacing || "";
  
    console.log(spacing + "* " + this.key);
    if (printData) { console.log(spacing + "* " + this.value); }
  
    if (!this.left && !this.right) { return; }
  
    if (this.left) {
      this.left.prettyPrint(printData, spacing + "  ");
    } else {
      console.log(spacing + "  *");
    }
    if (this.right) {
      this.right.prettyPrint(printData, spacing + "  ");
    } else {
      console.log(spacing + "  *");
    }
  }
}
