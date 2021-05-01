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
