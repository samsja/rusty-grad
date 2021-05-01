use rusty_grad::Variable;

#[cfg(test)]
fn test_main() {
    let mut x = Variable::new(1.0);
    println! {"{:}",x};

    let mut y = Variable::new(3.0);
    println! {"{:}",y};

    let mut sum = &mut x + &mut y;
    println!("{:?}", sum);

    sum.backward();

    for some_var in vec![&mut sum.left_root, &mut sum.right_root].iter_mut() {
        match some_var {
            Some(var) => println!("{}", var.grad),
            None => (),
        }
    }
}
