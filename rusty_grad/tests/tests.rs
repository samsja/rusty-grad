use rusty_grad::Variable;
use rusty_grad::VariableRef;

#[test]
fn test_main() {
    let x = VariableRef::new(Variable::new(1.0));
    let y = VariableRef::new(Variable::new(3.0));

    let mut sum = x.clone() + y.clone();

    sum.backward();

    println!("{}", x);
}