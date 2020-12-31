use crate::shape::*;

#[test]
fn test_size_of_rank_zero() {
    assert_eq!(get_size(&vec![]), 0);
}

#[test]
fn test_size_of_rank_one() {
    assert_eq!(get_size(&vec![5]), 5);
    assert_eq!(get_size(&vec![6]), 6);
    assert_eq!(get_size(&vec![7]), 7);
    assert_eq!(get_size(&vec![22]), 22);
    assert_eq!(get_size(&vec![33]), 33);
}

#[test]
fn test_size_of_higher_rank() {
    assert_eq!(get_size(&vec![5, 6, 7]), 210);
    assert_eq!(get_size(&vec![1, 2, 3]), 6);
    assert_eq!(get_size(&vec![1, 55, 2]), 110);
    assert_eq!(get_size(&vec![88, 72, 0]), 0);
}

#[test]
fn test_stride_of_rank_zero() {
    assert_eq!(compute_strides(&vec![]), vec![]);
}

#[test]
fn test_stride_of_rank_one() {
    assert_eq!(compute_strides(&vec![5]), vec![1]);
    assert_eq!(compute_strides(&vec![22]), vec![1]);
}

#[test]
fn test_stride_of_higher_rank() {
    assert_eq!(compute_strides(&vec![5, 2, 3]), vec![6,3,1]);
    assert_eq!(compute_strides(&vec![22, 5, 2, 3]), vec![30,6,3,1]);
    assert_eq!(compute_strides(&vec![22, 10, 5, 6, 3]), vec![900,90,18,3,1]);
}

#[test]
fn test_index_to_pos_rank_zero() {
    let shape = vec![];
    let strides = compute_strides(&shape);

    assert_eq!(index_to_pos(&vec![], &strides), 0);
}

#[test]
fn test_index_to_pos_rank_one() {
    let shape = vec![22];
    let strides = compute_strides(&shape);

    assert_eq!(index_to_pos(&vec![1], &strides), 1);
    assert_eq!(index_to_pos(&vec![5], &strides), 5);
    assert_eq!(index_to_pos(&vec![21], &strides), 21);
}

#[test]
fn test_index_to_pos_higher_rank() {
    let shape = vec![4,3,2];
    let strides = compute_strides(&shape);

    assert_eq!(index_to_pos(&vec![0, 1, 1], &strides), 3);
    assert_eq!(index_to_pos(&vec![1, 0, 1], &strides), 7);
    assert_eq!(index_to_pos(&vec![2,2,1], &strides), 17);
    assert_eq!(index_to_pos(&vec![3,2,1], &strides), 23);
}

#[test]
fn test_pos_to_index_rank_zero() {
    let shape = vec![];
    let strides = compute_strides(&shape);

    assert_eq!(pos_to_index(0, &strides), vec![]);
}

#[test]
fn test_pos_to_index_rank_one() {
    let shape = vec![22];
    let strides = compute_strides(&shape);

    assert_eq!(pos_to_index(1, &strides), vec![1]);
    assert_eq!(pos_to_index(5, &strides), vec![5]);
    assert_eq!(pos_to_index(21, &strides), vec![21]);
}

#[test]
fn test_pos_to_index_higher_rank() {
    let shape = vec![4,3,2];
    let strides = compute_strides(&shape);

    assert_eq!(pos_to_index(3, &strides), vec![0, 1, 1]);
    assert_eq!(pos_to_index(7, &strides), vec![1, 0, 1]);
    assert_eq!(pos_to_index(17, &strides), vec![2, 2, 1]);
    assert_eq!(pos_to_index(23, &strides), vec![3, 2, 1]);
}