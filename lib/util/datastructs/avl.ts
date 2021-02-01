import {BSTOptions, BinarySearchTree} from './bst';
import {OrderedDict, SearchQuery} from './types';

export class AVLTree<K, V> implements OrderedDict<K, V> {
  private tree: _AVLTree<K, V>;
  constructor(options: BSTOptions<K, V>) {
    this.tree = new _AVLTree(options);
  }

  checkIsAVLT() {
    this.tree.checkIsAVLT();
  }

  insert(key: K, value: V) {
    const newTree = this.tree.insert(key, value);

    // If newTree is undefined, that means its structure was not modified
    if (newTree) {
      this.tree = newTree;
    }
  }

  delete(key: K, value: V) {
    const newTree = this.tree.delete(key, value);

    // If newTree is undefined, that means its structure was not modified
    if (newTree) {
      this.tree = newTree;
    }
  }

  deleteFirst(key: K) {
    const newTree = this.tree.deleteFirst(key);

    // If newTree is undefined, that means its structure was not modified
    if (newTree) {
      this.tree = newTree;
    }
  }

  getNumberOfKeys() {
    return this.tree.getNumberOfKeys();
  }

  search(key: K) {
    return this.tree.search(key);
  }

  betweenBounds(query: SearchQuery<K>) {
    return this.tree.betweenBounds(query);
  }

  betweenBoundsFirst(query: SearchQuery<K>) {
    return this.tree.betweenBoundsFirst(query);
  }

  prettyPrint(printData: boolean, spacing?: string) {
    this.tree.prettyPrint(printData, spacing);
  }

  executeOnEveryNode(fn: (tree: _AVLTree<K, V>) => void) {
    //@ts-ignore
    this.tree.executeOnEveryNode(fn);
  }
}

class _AVLTree<K, V> extends BinarySearchTree<K, V> {
  protected left?: _AVLTree<K, V>;
  protected right?: _AVLTree<K, V>;
  protected parent?: _AVLTree<K, V>;

  protected height?: number;

  constructor(options: BSTOptions<K, V>) {
    super(options);
  }

  checkHeightCorrect() {
    if (this.key === undefined) {
      return;
    } // Empty tree

    if (this.left && this.left.height === undefined) {
      throw new Error('Undefined height for node ' + this.left.key);
    }
    if (this.right && this.right.height === undefined) {
      throw new Error('Undefined height for node ' + this.right.key);
    }
    if (this.height === undefined) {
      throw new Error('Undefined height for node ' + this.key);
    }

    const leftH = this.left && this.left.height ? this.left.height : 0;
    const rightH = this.right && this.right.height ? this.right.height : 0;

    if (this.height !== 1 + Math.max(leftH, rightH)) {
      throw new Error('Height constraint failed for node ' + this.key);
    }
    if (this.left) {
      this.left.checkHeightCorrect();
    }
    if (this.right) {
      this.right.checkHeightCorrect();
    }
  }

  balanceFactor() {
    const leftH = this.left && this.left.height ? this.left.height : 0;
    const rightH = this.right && this.right.height ? this.right.height : 0;
    return leftH - rightH;
  }

  checkBalanceFactors() {
    if (Math.abs(this.balanceFactor()) > 1) {
      throw new Error('Tree is unbalanced at node ' + this.key);
    }

    if (this.left) {
      this.left.checkBalanceFactors();
    }
    if (this.right) {
      this.right.checkBalanceFactors();
    }
  }

  checkIsAVLT() {
    super.checkIsBST();
    this.checkHeightCorrect();
    this.checkBalanceFactors();
  }

  rightRotation() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const q = this;
    const p = this.left;

    if (!p) {
      return this;
    }

    const b = p.right;

    // Alter tree structure
    if (q.parent) {
      p.parent = q.parent;
      if (q.parent.left === q) {
        q.parent.left = p;
      } else {
        q.parent.right = p;
      }
    } else {
      p.parent = undefined;
    }
    p.right = q;
    q.parent = p;
    q.left = b;
    if (b) {
      b.parent = q;
    }

    // Update heights
    const ah = p.left && p.left.height ? p.left.height : 0;
    const bh = b && b.height ? b.height : 0;
    const ch = q.right && q.right.height ? q.right.height : 0;
    q.height = Math.max(bh, ch) + 1;
    p.height = Math.max(ah, q.height) + 1;

    return p;
  }

  leftRotation() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const p = this;
    const q = this.right;

    if (!q) {
      return this;
    } // No change

    const b = q.left;

    // Alter tree structure
    if (p.parent) {
      q.parent = p.parent;
      if (p.parent.left === p) {
        p.parent.left = q;
      } else {
        p.parent.right = q;
      }
    } else {
      q.parent = undefined;
    }
    q.left = p;
    p.parent = q;
    p.right = b;
    if (b) {
      b.parent = p;
    }

    // Update heights
    const ah = p.left ? p.left.height : 0;
    const bh = b ? b.height : 0;
    const ch = q.right ? q.right.height : 0;
    //@ts-ignore
    p.height = Math.max(ah, bh) + 1;
    //@ts-ignore
    q.height = Math.max(ch, p.height) + 1;

    return q;
  }

  rightTooSmall() {
    if (this.balanceFactor() <= 1) {
      return this;
    } // Right is not too small, don't change

    if (this.left && this.left.balanceFactor() < 0) {
      this.left.leftRotation();
    }

    return this.rightRotation();
  }

  leftTooSmall() {
    if (this.balanceFactor() >= -1) {
      return this;
    } // Left is not too small, don't change

    if (this.right && this.right.balanceFactor() > 0) {
      this.right.rightRotation();
    }

    return this.leftRotation();
  }

  rebalanceAlongPath(path: _AVLTree<K, V>[]) {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let newRoot: _AVLTree<K, V> = this;
    let rotated;
    let i;

    if (this.key === undefined) {
      delete this.height;
      return this;
    } // Empty tree

    // Rebalance the tree and update all heights
    for (i = path.length - 1; i >= 0; i -= 1) {
      path[i].height =
        1 +
        Math.max(
          //@ts-ignore
          path[i].left ? path[i].left.height : 0,
          //@ts-ignore
          path[i].right ? path[i].right.height : 0
        );

      if (path[i].balanceFactor() > 1) {
        rotated = path[i].rightTooSmall();
        if (i === 0) {
          newRoot = rotated;
        }
      }

      if (path[i].balanceFactor() < -1) {
        rotated = path[i].leftTooSmall();
        if (i === 0) {
          newRoot = rotated;
        }
      }
    }

    return newRoot;
  }

  insert(key: K, value: V) {
    const insertPath: _AVLTree<K, V>[] = [];
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let currentNode: _AVLTree<K, V> = this;

    // Empty tree, insert as root
    if (this.key === undefined) {
      this.key = key;
      this.value.push(value);
      this.height = 1;
      return this;
    }

    // Insert new leaf at the right place
    // eslint-disable-next-line no-constant-condition
    while (true) {
      // Same key: no change in the tree structure
      //@ts-ignore
      if (currentNode.compareKeys(currentNode.key, key) === 0) {
        if (currentNode.unique) {
          const err = new Error(
            "Can't insert key " + key + ', it violates the unique constraint'
          );
          throw err;
        } else {
          currentNode.value.push(value);
        }
        return this;
      }

      insertPath.push(currentNode);

      //@ts-ignore
      if (currentNode.compareKeys(key, currentNode.key) < 0) {
        if (!currentNode.left) {
          insertPath.push(
            currentNode.createLeftChild({key: key, value: value}) as _AVLTree<
              K,
              V
            >
          );
          break;
        } else {
          currentNode = currentNode.left;
        }
      } else {
        if (!currentNode.right) {
          insertPath.push(
            currentNode.createRightChild({key: key, value: value}) as _AVLTree<
              K,
              V
            >
          );
          break;
        } else {
          currentNode = currentNode.right;
        }
      }
    }

    return this.rebalanceAlongPath(insertPath);
  }

  createSimilar(options: BSTOptions<K, V>): _AVLTree<K, V> {
    options = options || {};
    options.unique = this.unique;
    options.compareKeys = this.compareKeys;

    return new _AVLTree(options);
  }

  delete(key: K, value: V) {
    const newData: V[] = [];
    let replaceWith;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let currentNode: _AVLTree<K, V> = this;
    const deletePath = [];

    if (this.key === undefined) {
      return this;
    } // Empty tree

    // Either no match is found and the function will return from within the loop
    // Or a match is found and deletePath will contain the path from the root to the node to delete after the loop
    // eslint-disable-next-line no-constant-condition
    while (true) {
      //@ts-ignore
      if (currentNode.compareKeys(key, currentNode.key) === 0) {
        break;
      }

      deletePath.push(currentNode);

      //@ts-ignore
      if (currentNode.compareKeys(key, currentNode.key) < 0) {
        if (currentNode.left) {
          currentNode = currentNode.left;
        } else {
          return this; // Key not found, no modification
        }
      } else {
        // currentNode.compareKeys(key, currentNode.key) is > 0
        if (currentNode.right) {
          currentNode = currentNode.right;
        } else {
          return this; // Key not found, no modification
        }
      }
    }

    // Delete only a value (no tree modification)
    if (currentNode.value.length > 1 && value !== undefined) {
      currentNode.value.forEach(d => {
        if (!currentNode.compareValues(d, value)) {
          newData.push(d);
        }
      });
      currentNode.value = newData;
      return this;
    }

    // Delete a whole node

    // Leaf
    if (!currentNode.left && !currentNode.right) {
      if (currentNode === this) {
        // This leaf is also the root
        delete currentNode.key;
        currentNode.value = [];
        delete currentNode.height;
        return this;
      } else {
        //@ts-ignore
        if (currentNode.parent.left === currentNode) {
          //@ts-ignore
          currentNode.parent.left = undefined;
        } else {
          //@ts-ignore
          currentNode.parent.right = undefined;
        }
        return this.rebalanceAlongPath(deletePath);
      }
    }

    // Node with only one child
    if (!currentNode.left || !currentNode.right) {
      replaceWith = currentNode.left ? currentNode.left : currentNode.right;

      if (currentNode === this) {
        // This node is also the root
        //@ts-ignore
        replaceWith.parent = undefined;
        return replaceWith; // height of replaceWith is necessarily 1 because the tree was balanced before deletion
      } else {
        //@ts-ignore
        if (currentNode.parent.left === currentNode) {
          //@ts-ignore
          currentNode.parent.left = replaceWith;
          //@ts-ignore
          replaceWith.parent = currentNode.parent;
        } else {
          //@ts-ignore
          currentNode.parent.right = replaceWith;
          //@ts-ignore
          replaceWith.parent = currentNode.parent;
        }

        return this.rebalanceAlongPath(deletePath);
      }
    }

    // Node with two children
    // Use the in-order predecessor (no need to randomize since we actively rebalance)
    deletePath.push(currentNode);
    replaceWith = currentNode.left;

    // Special case: the in-order predecessor is right below the node to delete
    if (!replaceWith.right) {
      currentNode.key = replaceWith.key;
      currentNode.value = replaceWith.value;
      currentNode.left = replaceWith.left;
      if (replaceWith.left) {
        replaceWith.left.parent = currentNode;
      }
      return this.rebalanceAlongPath(deletePath);
    }

    // After this loop, replaceWith is the right-most leaf in the left subtree
    // and deletePath the path from the root (inclusive) to replaceWith (exclusive)
    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (replaceWith.right) {
        deletePath.push(replaceWith);
        replaceWith = replaceWith.right;
      } else {
        break;
      }
    }

    currentNode.key = replaceWith.key;
    currentNode.value = replaceWith.value;

    //@ts-ignore
    replaceWith.parent.right = replaceWith.left;
    if (replaceWith.left) {
      replaceWith.left.parent = replaceWith.parent;
    }

    return this.rebalanceAlongPath(deletePath);
  }

  deleteFirst(key: K) {
    let replaceWith;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let currentNode: _AVLTree<K, V> = this;
    const deletePath = [];

    if (this.key === undefined) {
      return this;
    } // Empty tree

    // Either no match is found and the function will return from within the loop
    // Or a match is found and deletePath will contain the path from the root to the node to delete after the loop
    // eslint-disable-next-line no-constant-condition
    while (true) {
      //@ts-ignore
      if (currentNode.compareKeys(key, currentNode.key) === 0) {
        break;
      }

      deletePath.push(currentNode);

      //@ts-ignore
      if (currentNode.compareKeys(key, currentNode.key) < 0) {
        if (currentNode.left) {
          currentNode = currentNode.left;
        } else {
          return this; // Key not found, no modification
        }
      } else {
        // currentNode.compareKeys(key, currentNode.key) is > 0
        if (currentNode.right) {
          currentNode = currentNode.right;
        } else {
          return this; // Key not found, no modification
        }
      }
    }

    // Delete only a value (no tree modification)
    if (currentNode.value.length > 1) {
      currentNode.value = currentNode.value.slice(1);
      return this;
    }

    // Delete a whole node

    // Leaf
    if (!currentNode.left && !currentNode.right) {
      if (currentNode === this) {
        // This leaf is also the root
        delete currentNode.key;
        currentNode.value = [];
        delete currentNode.height;
        return this;
      } else {
        //@ts-ignore
        if (currentNode.parent.left === currentNode) {
          //@ts-ignore
          currentNode.parent.left = null;
        } else {
          //@ts-ignore
          currentNode.parent.right = null;
        }
        return this.rebalanceAlongPath(deletePath);
      }
    }

    // Node with only one child
    if (!currentNode.left || !currentNode.right) {
      replaceWith = currentNode.left ? currentNode.left : currentNode.right;

      if (currentNode === this) {
        // This node is also the root
        //@ts-ignore
        replaceWith.parent = null;
        return replaceWith; // height of replaceWith is necessarily 1 because the tree was balanced before deletion
      } else {
        //@ts-ignore
        if (currentNode.parent.left === currentNode) {
          //@ts-ignore
          currentNode.parent.left = replaceWith;
          //@ts-ignore
          replaceWith.parent = currentNode.parent;
        } else {
          //@ts-ignore
          currentNode.parent.right = replaceWith;
          //@ts-ignore
          replaceWith.parent = currentNode.parent;
        }

        return this.rebalanceAlongPath(deletePath);
      }
    }

    // Node with two children
    // Use the in-order predecessor (no need to randomize since we actively rebalance)
    deletePath.push(currentNode);
    replaceWith = currentNode.left;

    // Special case: the in-order predecessor is right below the node to delete
    if (!replaceWith.right) {
      currentNode.key = replaceWith.key;
      currentNode.value = replaceWith.value;
      currentNode.left = replaceWith.left;
      if (replaceWith.left) {
        replaceWith.left.parent = currentNode;
      }
      return this.rebalanceAlongPath(deletePath);
    }

    // After this loop, replaceWith is the right-most leaf in the left subtree
    // and deletePath the path from the root (inclusive) to replaceWith (exclusive)
    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (replaceWith.right) {
        deletePath.push(replaceWith);
        replaceWith = replaceWith.right;
      } else {
        break;
      }
    }

    currentNode.key = replaceWith.key;
    currentNode.value = replaceWith.value;

    //@ts-ignore
    replaceWith.parent.right = replaceWith.left;
    if (replaceWith.left) {
      replaceWith.left.parent = replaceWith.parent;
    }

    return this.rebalanceAlongPath(deletePath);
  }
}
