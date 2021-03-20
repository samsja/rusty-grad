mod variable;

use variable::Variable;

fn main() {
    let x = Variable::new(1.0);
    println! {"{:}",x};

    let y = Variable::new(3.0);
    println! {"{:}",y};

    let sum = &x + &y;
    println!("{:?}", sum);

    let z = Variable::new_node(1.0, Some(&sum), None);
    println! {"{:?}",z};
}
