use crate::module::Module;
use crate::variable::{Variable, VariableRef};
use ndarray::{Array, Ix2, NdFloat};

//use rand::distributions::{Uniform};

pub struct Linear<T: NdFloat> {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: VariableRef<T>,
    pub bias: VariableRef<T>,
}

impl<T: NdFloat> Linear<T> {
    pub fn new(in_features: usize, out_features: usize) -> Linear<T> {
        Linear {
            in_features,
            out_features,
            weight: Linear::init_weight(in_features, out_features),
            bias: Linear::init_bias(out_features),
        }
    }

    fn init_weight(in_features: usize, out_features: usize) -> VariableRef<T> {
        //let bound = 1. / (in_features as f32).sqrt() ;
        // let between = Uniform::from(-bound..bound);
        let array_init = Array::<T, Ix2>::ones((out_features, in_features));
        Variable::new(array_init.into_dyn())
    }

    fn init_bias(out_features: usize) -> VariableRef<T> {
        Variable::new(Array::<T, Ix2>::zeros((out_features, 1)).into_dyn())
    }
}

impl<T: NdFloat> Module<T> for Linear<T> {
    fn params(&self) -> Vec<VariableRef<T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn f(&mut self, input: &VariableRef<T>) -> VariableRef<T> {
        self.weight.dot(&input) + self.bias.clone()
    }
}

pub struct MLP<T: NdFloat> {
    pub layers: Vec<Linear<T>>,
}

impl<T: NdFloat> Module<T> for MLP<T> {
    fn params(&self) -> Vec<VariableRef<T>> {
        let mut params: Vec<VariableRef<T>> = vec![];

        for lay in self.layers.iter() {
            Vec::append(&mut params, &mut lay.params().clone());
        }

        params
    }

    fn f(&mut self, input: &VariableRef<T>) -> VariableRef<T> {
        let mut output = input.clone();
        for i in 0..(self.layers.len() - 1) {
            output = self.layers[i].f(&output).relu();
        }

        let len = self.layers.len();

        self.layers[len - 1].f(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_forward() {
        let ref x = Variable::new(Array::<f32, _>::ones((3, 1)).into_dyn());

        let mut layer = Linear::new(3, 2);
        layer.params();

        let mut y = layer.f(x);

        y.backward();

        let out = Array::<f32, _>::zeros((2, 1));
        let shape = out.shape();
        assert_eq!(shape, y.borrow().data.shape());
    }

    #[test]
    fn mlp_forward() {
        let layer1 = Linear::new(5, 10);
        let layer2 = Linear::new(10, 10);
        let layer3 = Linear::new(10, 2);

        let mut mlp = MLP {
            layers: vec![layer1, layer2, layer3],
        };
        mlp.params();

        let ref x = Variable::new(Array::<f32, Ix2>::zeros((5, 1)).into_dyn());

        let mut y = mlp.f(x);

        let out = Array::<f32, _>::zeros((2, 1));
        let shape = out.shape();

        y.backward();

        assert_eq!(shape, y.borrow().data.shape());
    }
}
