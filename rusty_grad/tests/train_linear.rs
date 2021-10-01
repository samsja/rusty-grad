use ndarray::{array, Array};
use rusty_grad::grad_fn::functional::loss::mse_loss;
use rusty_grad::module::Module;
use rusty_grad::nn::linear::{Linear, MLP};
use rusty_grad::optim::sgd::{Optim, SGD};
use rusty_grad::variable::Variable;

#[test]
fn mlp_train() {
    let data = Variable::new_no_retain_grad(array!([1.0], [1.0]).into_dyn());

    let layer1 = Linear::<f32>::new(2, 10);
    let layer2 = Linear::<f32>::new(10, 10);
    let layer3 = Linear::<f32>::new(10, 2);

    let mut mlp = MLP {
        layers: vec![layer1, layer2, layer3],
    };

    let ref mut optim = SGD::new(mlp.params(), 0.01).unwrap();

    for _epoch in 0..2 {
        mlp.zero_grad();

        let output = mlp.f(&data);
        let target =
            Variable::new_no_retain_grad(Array::<f32, _>::zeros(output.borrow().data.shape()));
        let mut loss = mse_loss(&output, &target);

        loss.backward();
        optim.step();
    }
}
