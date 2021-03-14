mod variable;

use variable::Variable;


fn main() {


    let x = Variable::new(1.0);
    let y = Variable::new(2.0);

    let sum = &x+&y;

    println!("{:}", sum);
}
