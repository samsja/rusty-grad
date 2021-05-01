use rusty_grad::Variable;
use rusty_grad::VariableRef;

#[test]
fn test_simple_autograd() {
    let x = VariableRef::new(Variable::new(1.0));
    let y = VariableRef::new(Variable::new(3.0));

    let mut sum = x.clone() + y.clone();

    sum.backward();

    assert_eq!(x.borrow().grad, 1.0);
    assert_eq!(y.borrow().grad, 3.0);
}

#[test]
fn test_simple_autograd_with_const() {
    let x_ = VariableRef::new(Variable::new(4.0));
    let y_ = VariableRef::new(Variable::new(3.0));

    let x = x_.clone();
    let y = y_.clone();

    let mut z = z = (x + 2) / (y - 4);

    z.backward();

    assert_eq!(x_.borrow().grad, -1.0);
    assert_eq!(y_.borrow().grad, 6.0);
}

#[test]
fn test_complex_autograd() {
    let x_ = VariableRef::new(Variable::new(4.0));
    let y_ = VariableRef::new(Variable::new(3.0));

    let x = x_.clone();
    let y = y_.clone();

    let mut z = (x + x * x + y) / (x * y + x);

    z.backward();

    assert_eq!(x_.borrow().grad, 0.203125);
    assert_eq!(y_.borrow().grad, -0.296875);
}
