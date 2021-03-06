use rusty_grad::variable::Variable;
use rusty_grad::variable::VariableRef;

#[test]
fn test_double_add() {
    let ref x = Variable::new(4.0);
    let mut z = x + x;

    z.backward();

    assert_eq!(x.borrow().grad, 2.0);
}

#[test]
fn test_simple_autograd() {
    let ref x = Variable::new(4.0);
    let ref y = Variable::new(3.0);

    let mut z = (x + x) + (x + y);

    z.backward();

    assert_eq!(x.borrow().grad, 3.0);
    assert_eq!(y.borrow().grad, 1.0);
}

#[test]
fn test_simple_two_stage_autograd() {
    let ref x = Variable::new(3.0);
    let ref y = Variable::new(5.0);

    let ref h = x + y;
    let mut z = h * x;

    z.backward();

    assert_eq!(x.borrow().grad, 11.0);
    assert_eq!(y.borrow().grad, 3.0);
}

#[test]
fn test_complexautograd_1() {
    let ref x = Variable::new(8.0);
    let ref y = Variable::new(-3.0);

    let mut z = (x * y) * (x * y) + (x - y);

    z.backward();

    assert_eq!(x.borrow().grad, 145.0);
    assert_eq!(y.borrow().grad, -385.0);
}

#[test]
fn test_complexautograd_2() {
    let ref x = Variable::new(-8.0);
    let ref y = Variable::new(13.0);

    let mut z = (x + y) * (x + y);
    z.backward();

    assert_eq!(x.borrow().grad, 10.0);
    assert_eq!(y.borrow().grad, 10.0);
}
