use rusty_grad::Variable;

#[test]
fn test_main() {
    let mut x_val = Variable::new(1.0);
    let x = &mut x_val;

    let mut y_val = Variable::new(3.0);
    let y = &mut y_val;

    let mut sum = x + y;

    sum.backward();

    for some_var in vec![&mut sum.left_root, &mut sum.right_root].iter_mut() {
        match some_var {
            Some(var) => println!("{}", var.grad),
            None => (),
        }
    }
}
