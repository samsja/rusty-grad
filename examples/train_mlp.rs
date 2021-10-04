use ndarray::{array, Array, Ix1};

use rusty_grad::data::moon::MakeMoonDataset;
use rusty_grad::grad_fn::functional::loss::mse_loss;
use rusty_grad::module::Module;
use rusty_grad::nn::linear::{Linear, MLP};
use rusty_grad::optim::sgd::{Optim, SGD};
use rusty_grad::variable::Variable;

fn main() {
    let dataset = MakeMoonDataset::new(30);

    let layer1 = Linear::<f32>::new(2, 16);
    let layer2 = Linear::<f32>::new(16, 2);

    let mut mlp = MLP {
        layers: vec![layer1, layer2],
    };

    let ref mut optim = SGD::new(mlp.params(), 0.3).unwrap();

    let period_print = 1;

    let mut loss = Variable::new(Array::<f32, Ix1>::zeros(1).into_dyn());

    for epoch in 0..1000 {
        mlp.zero_grad();

        for idx in 0..dataset.len() {
            let data_n_label = dataset.get(idx);
            let (data, label) = data_n_label;

            let output = mlp.f(&data).softmax();

            let target = if label == 0. {
                Variable::new_no_retain_grad(array!([0.], [1.]).into_dyn())
            } else {
                Variable::new_no_retain_grad(array!([1.], [0.]).into_dyn())
            };

            loss = loss + mse_loss(&output, &target);
        }

        loss = loss / Variable::new((dataset.len() as f32) * Array::<f32, Ix1>::ones(1).into_dyn());

        loss.backward();
        optim.step();

        if epoch % period_print == 0 {
            println!("epoch {} : loss : {}", epoch, loss);
        }
    }
}
