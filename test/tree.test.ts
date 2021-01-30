import { AVLTree } from "../lib/util/datastructs/avl";

describe('Create and insert', () => {
  it('should have zero keys in the beginning', () => {
    const tree = new AVLTree<number, number>({});

    expect(tree.getNumberOfKeys()).toBe(0);
  });

  it('should have keys after insertions', () => {
    const tree = new AVLTree<number, number>({});

    expect(tree.getNumberOfKeys()).toBe(0);

    tree.insert(5, 2);
    expect(tree.getNumberOfKeys()).toBe(1);

    tree.insert(6, 2);
    expect(tree.getNumberOfKeys()).toBe(2);

    tree.insert(4, 2);
    expect(tree.getNumberOfKeys()).toBe(3);

    tree.insert(4.5, 2);
    expect(tree.getNumberOfKeys()).toBe(4);

    tree.delete(6, 2);
    expect(tree.getNumberOfKeys()).toBe(3);
  });

  it('should have keys after insertions', () => {
    const tree = new AVLTree<number, number>({});

    expect(tree.getNumberOfKeys()).toBe(0);

    const inserts = [1,5,7,2,4,3,9,6,2,7];

    for (let i = 0; i < inserts.length; i++) {
      tree.insert(inserts[i], i);
    }

    expect(tree.getNumberOfKeys()).toBe(8);
    expect(tree.search(7)).toEqual([2,9]);
    expect(tree.betweenBounds({ gte: 2, lte: 5})).toEqual([
      {key: 2, value: 3},
      {key: 2, value: 8},
      {key: 3, value: 5},
      {key: 4, value: 4},
      {key: 5, value: 1}
    ]);

    const deletes = [2,5,7];
    for (let i = 0; i < deletes.length; i++) {
      const found = tree.search(deletes[i]);
      for (let j = 0; j < found.length; j++) {
        tree.delete(deletes[i], found[j]);
      }
    }

    expect(tree.getNumberOfKeys()).toBe(5);
  });

  for (let i = 0; i < 5; i++) {
    it(`should have keys after insertions ${i}`, () => {
      const tree = new AVLTree<number, number>({});
  
      const keys = [];
      const values = [];
      for (let i = 0; i < 1000; i++) {
        const key = Math.random();
        const value = Math.random();
  
        keys.push(key);
        values.push(value);
  
        tree.insert(key, value);
      }
  
      expect(tree.getNumberOfKeys()).toBe(keys.length);
  
      for (let i = 0; i < 250; i++) {
        tree.delete(keys[i], values[i]);
      }
  
      expect(tree.getNumberOfKeys()).toBe(keys.length - 250);
  
      for (let i = 0; i < 250; i++) {
        const key = Math.random();
        const value = Math.random();
  
        tree.insert(key, value);
      }
      expect(tree.getNumberOfKeys()).toBe(1000);
  
      let query = { gte: 0.25, lte: 0.75 };
      let result = tree.betweenBounds(query);
      
      expect(result.length).toBeLessThan(650);
      expect(result.length).toBeGreaterThan(350);
  
      query = { gte: 0.125, lte: 0.325 };
      result = tree.betweenBounds(query);
      
      expect(result.length).toBeLessThan(400);
      expect(result.length).toBeGreaterThan(100);
    });
  }
});
