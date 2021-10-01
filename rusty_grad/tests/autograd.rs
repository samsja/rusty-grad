use ndarray::array;
use rusty_grad::modules::functional::loss::mse_loss;
use rusty_grad::variable::Variable;

#[test]
fn test_double_add() {
    let ref x = Variable::new(array!([4.0]).into_dyn());
    let mut z = x + x;

    z.backward();

    assert_eq!(x.borrow().get_grad_f(), array!([2.0]).into_dyn());
}

#[test]
fn test_simple_autograd() {
    let ref x = Variable::new(array!([4.0]).into_dyn());
    let ref y = Variable::new(array!([3.0]).into_dyn());

    let mut z = (x + x) + (x + y);

    z.backward();

    assert_eq!(x.borrow().get_grad_f(), array!([3.0]).into_dyn());
    assert_eq!(y.borrow().get_grad_f(), array!([1.0]).into_dyn());
}

#[test]
fn test_simple_two_stage_autograd() {
    let ref x = Variable::new(array!([3.0]).into_dyn());
    let ref y = Variable::new(array!([5.0]).into_dyn());

    let ref h = x + y;
    let mut z = h * x;

    z.backward();

    assert_eq!(x.borrow().get_grad_f(), array!([11.0]).into_dyn());
    assert_eq!(y.borrow().get_grad_f(), array!([3.0]).into_dyn());
}

#[test]
fn test_complexautograd_1() {
    let ref x = Variable::new(array!([8.0]).into_dyn());
    let ref y = Variable::new(array!([-3.0]).into_dyn());

    let mut z = (x * y) * (x * y) + (x - y);

    z.backward();

    assert_eq!(x.borrow().get_grad_f(), array!([145.0]).into_dyn());
    assert_eq!(y.borrow().get_grad_f(), array!([-385.0]).into_dyn());
}

#[test]
fn test_complexautograd_2() {
    let ref x = Variable::new(array!([-8.0]).into_dyn());
    let ref y = Variable::new(array!([13.0]).into_dyn());

    let mut z = (x + y) * (x + y);
    z.backward();

    assert_eq!(x.borrow().get_grad_f(), array!([10.0]).into_dyn());
    assert_eq!(y.borrow().get_grad_f(), array!([10.0]).into_dyn());
}

#[test]
fn test_mat_vect_mse() {
    let mut x = Variable::new(array!([1.0, 2.0], [3.0, 4.0]).into_dyn());
    let y = Variable::new(array!([1.0], [2.0]).into_dyn());

    let zero = Variable::new(array!([0.0], [0.0]).into_dyn());

    let z = &x.dot(&y);
    let mut h = mse_loss(z, &zero);
    h.backward();

    println!("z {}", z);
    println!("h {}", h);

    assert_eq!(
        x.borrow().get_grad_f(),
        array!([5.0, 10.0], [11.0, 22.0]).into_dyn()
    );
    assert_eq!(y.borrow().get_grad_f(), array!([38.0], [54.0]).into_dyn());
}
