use ndarray::array;
use rusty_grad::variable::Variable;

#[test]
fn test_double_add() {
    let ref x = Variable::new(array!([4.0]));
    let mut z = x + x;

    z.backward();

    assert_eq!(x.borrow().get_grad(), array!([2.0]));
}

#[test]
fn test_simple_autograd() {
    let ref x = Variable::new(array!([4.0]));
    let ref y = Variable::new(array!([3.0]));

    let mut z = (x + x) + (x + y);

    z.backward();

    assert_eq!(x.borrow().get_grad(), array!([3.0]));
    assert_eq!(y.borrow().get_grad(), array!([1.0]));
}

#[test]
fn test_simple_two_stage_autograd() {
    let ref x = Variable::new(array!([3.0]));
    let ref y = Variable::new(array!([5.0]));

    let ref h = x + y;
    let mut z = h * x;

    z.backward();

    assert_eq!(x.borrow().get_grad(), array!([11.0]));
    assert_eq!(y.borrow().get_grad(), array!([3.0]));
}

#[test]
fn test_complexautograd_1() {
    let ref x = Variable::new(array!([8.0]));
    let ref y = Variable::new(array!([-3.0]));

    let mut z = (x * y) * (x * y) + (x - y);

    z.backward();

    assert_eq!(x.borrow().get_grad(), array!([145.0]));
    assert_eq!(y.borrow().get_grad(), array!([-385.0]));
}

#[test]
fn test_complexautograd_2() {
    let ref x = Variable::new(array!([-8.0]));
    let ref y = Variable::new(array!([13.0]));

    let mut z = (x + y) * (x + y);
    z.backward();

    assert_eq!(x.borrow().get_grad(), array!([10.0]));
    assert_eq!(y.borrow().get_grad(), array!([10.0]));
}
